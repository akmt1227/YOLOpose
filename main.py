import os
import json
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from collections import deque, Counter
from ultralytics import YOLO
import warnings
from models.anomaly_lstm import PoseActionLSTM
from utils.pose import normalize_keypoints, sample_interval, SEQ_LEN, TARGET_FPS
from utils.config import (YOLO_MODEL, CLASSES, CLASS_DISPLAY,
                          STALE_FRAMES, PRUNE_INTERVAL, SMOOTH_WINDOW, NG_HOLD_FRAMES)

warnings.filterwarnings('ignore')


def resolve_threshold(threshold):
    """NG probability threshold: a person is NG if P(NG) = 1 - P(normal) > threshold.

    Priority: explicit --threshold > calibrated threshold.json (from training) > 0.5.
    """
    if threshold is not None:
        print(f"Using NG threshold from --threshold: {threshold:.3f}")
        return threshold
    if os.path.exists("threshold.json"):
        with open("threshold.json") as f:
            threshold = float(json.load(f).get("threshold", 0.5))
        print(f"Loaded calibrated NG threshold from threshold.json: {threshold:.3f}")
        return threshold
    print("threshold.json not found; using default NG threshold 0.5.")
    return 0.5


def draw_ng_banner(frame, text):
    """Full-width red NG alert banner (with the NG reason) across the top."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 255), -1)
    cv2.putText(frame, text, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)


def process_video(input_path, output_path, model_name=YOLO_MODEL, threshold=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load YOLO Pose Model
    print(f"Loading {model_name}...")
    yolo_model = YOLO(model_name)
    yolo_model.to(device)

    # 2. Load our LSTM classifier
    print("Initializing LSTM classifier...")
    lstm_model = PoseActionLSTM().to(device)
    if os.path.exists("lstm_model.pth"):
        print("Loading trained weights from lstm_model.pth...")
        lstm_model.load_state_dict(torch.load("lstm_model.pth", map_location=device))
    else:
        print("Warning: lstm_model.pth not found. Using random weights! Train first (prepare_data.py -> train.py).")
    lstm_model.eval()

    threshold = resolve_threshold(threshold)

    # 3. Video IO setup
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {input_path}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30
    if width == 0 or height == 0:
        print(f"Error: Invalid video dimensions {width}x{height} for {input_path}")
        cap.release()
        return
    sample_every = sample_interval(fps)   # classify at ~TARGET_FPS, matching prepare_data.py

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    seq_len = SEQ_LEN
    # Per-tracked-person state
    history = {}      # id -> deque of pose feature vectors (one cycle)
    ng_votes = {}     # id -> deque of recent raw NG decisions (majority-vote smoothing)
    type_votes = {}   # id -> deque of recent NG type slugs (mode -> displayed reason)
    person_state = {} # id -> {'ng': bool, 'type': slug or None, 'peak_back': seconds}
    last_seen = {}    # id -> last frame index seen (for pruning)

    print(f"Starting video processing. Input: {input_path}, Output: {output_path}")
    frame_count = 0
    ng_hold = 0             # frames remaining to keep the NG banner visible
    banner_text = "NG"

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Track every frame (ID continuity + smooth skeleton overlay)...
        results = yolo_model.track(frame, persist=True, verbose=False)
        # ...but only run the classifier on sampled frames (matches prepare_data.py).
        sampled = (frame_count % sample_every == 0)

        annotated_frame = frame.copy()
        ng_now = None   # (type_slug, peak_back) of a currently-NG person, if any

        if results and len(results[0].boxes) > 0:
            result = results[0]
            annotated_frame = result.plot()  # draw skeletons/boxes

            if result.keypoints is not None and result.boxes.id is not None:
                keypoints = result.keypoints.xy.cpu().numpy()
                kconf = result.keypoints.conf
                kconf = kconf.cpu().numpy() if kconf is not None else np.ones(keypoints.shape[:2], dtype=np.float32)
                boxes = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.int().cpu().numpy()

                for i, track_id in enumerate(track_ids):
                    if sampled:
                        if track_id not in history:
                            history[track_id] = deque(maxlen=seq_len)
                            ng_votes[track_id] = deque(maxlen=SMOOTH_WINDOW)
                            type_votes[track_id] = deque(maxlen=SMOOTH_WINDOW)
                        last_seen[track_id] = frame_count

                        feat = normalize_keypoints(keypoints[i], kconf[i], boxes[i], (width, height))
                        history[track_id].append(feat)

                        # Once we have a full cycle, classify OK vs the NG types.
                        if len(history[track_id]) == seq_len:
                            seq_tensor = torch.tensor(np.array(history[track_id]),
                                                      dtype=torch.float32).unsqueeze(0).to(device)
                            with torch.no_grad():
                                logits, attn = lstm_model(seq_tensor, return_attn=True)
                                probs = F.softmax(logits, dim=1)[0]
                            p_ng = 1.0 - probs[0].item()
                            raw_ng = p_ng > threshold

                            ng_votes[track_id].append(raw_ng)
                            if raw_ng:
                                # Most likely NG type + when in the window it happened
                                ng_idx = 1 + int(torch.argmax(probs[1:]).item())
                                type_votes[track_id].append(CLASSES[ng_idx])

                            votes = ng_votes[track_id]
                            is_ng = sum(votes) > len(votes) / 2
                            ng_type, peak_back = None, None
                            if is_ng and len(type_votes[track_id]) > 0:
                                ng_type = Counter(type_votes[track_id]).most_common(1)[0][0]
                                peak_idx = int(torch.argmax(attn[0]).item())
                                peak_back = (seq_len - 1 - peak_idx) / TARGET_FPS
                            person_state[track_id] = {'ng': is_ng, 'type': ng_type,
                                                      'peak_back': peak_back}

                    # Draw the last known label for this person every frame.
                    state = person_state.get(track_id)
                    if state is not None:
                        if state['ng'] and state['type'] is not None:
                            ng_now = (state['type'], state['peak_back'])
                        x1, y1, x2, y2 = map(int, boxes[i])
                        label = "NG" if state['ng'] else "OK"
                        color = (0, 0, 255) if state['ng'] else (0, 255, 0)
                        cv2.rectangle(annotated_frame, (x1, max(0, y1 - 40)), (x1 + 80, max(0, y1)), color, -1)
                        cv2.putText(annotated_frame, label, (x1 + 5, max(0, y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # NG alert: raise/refresh a held banner (with reason) while anyone is NG.
        if ng_now is not None:
            ng_type, peak_back = ng_now
            banner_text = CLASS_DISPLAY.get(ng_type, "NG")
            if ng_hold == 0:
                t = frame_count / fps
                where = f", anomaly focus ~{peak_back:.1f}s ago in the window" if peak_back is not None else ""
                print(f"  [NG] {ng_type} at frame {frame_count} (~{t:.1f}s){where}", flush=True)
            ng_hold = NG_HOLD_FRAMES
        if ng_hold > 0:
            draw_ng_banner(annotated_frame, banner_text)
            ng_hold -= 1

        # Bound memory on long videos: drop tracks that have disappeared.
        if frame_count % PRUNE_INTERVAL == 0:
            for tid in [t for t, fs in last_seen.items() if frame_count - fs > STALE_FRAMES]:
                for d in (history, ng_votes, type_votes, person_state, last_seen):
                    d.pop(tid, None)

        out.write(annotated_frame)
        frame_count += 1
        if frame_count % 30 == 0:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"  Progress: {frame_count}/{total} frames ({(frame_count/max(1, total))*100:.1f}%)", flush=True)

    cap.release()
    out.release()
    print(f"Finished processing! Video saved to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="YOLO Pose + LSTM work-motion NG detection")
    parser.add_argument('--input', type=str, default='input.mp4', help='Path to input MP4 video')
    parser.add_argument('--output', type=str, default='output.mp4', help='Path to output MP4 video')
    parser.add_argument('--model', type=str, default=YOLO_MODEL, help='YOLO-Pose weights to use')
    parser.add_argument('--threshold', type=float, default=None,
                        help='NG threshold: flag when P(NG) = 1 - P(normal) > threshold. '
                             'Defaults to the calibrated value in threshold.json, else 0.5.')
    args = parser.parse_args()

    process_video(args.input, args.output, model_name=args.model, threshold=args.threshold)
