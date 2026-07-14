import os
import json
import cv2
import torch
import numpy as np
from collections import deque
from ultralytics import YOLO
from sentence_transformers import SentenceTransformer
import warnings
from models.anomaly_lstm import PoseActionLSTM
from utils.pose import normalize_keypoints, SEQ_LEN
from utils.config import YOLO_MODEL, STALE_FRAMES, PRUNE_INTERVAL, SMOOTH_WINDOW

warnings.filterwarnings('ignore')


def resolve_margin(margin):
    """Decide the abnormal-vs-normal decision margin.

    Priority: explicit --margin > calibrated threshold.json (written by training) > 0.0.
    The decision is: abnormal if (sim_abnormal - sim_normal) > margin.
    """
    if margin is not None:
        print(f"Using decision margin from --margin: {margin:.4f}")
        return margin
    if os.path.exists("threshold.json"):
        with open("threshold.json") as f:
            margin = float(json.load(f).get("margin", 0.0))
        print(f"Loaded calibrated decision margin from threshold.json: {margin:.4f}")
        return margin
    print("threshold.json not found; using default decision margin 0.0.")
    return 0.0


def process_video(input_path, output_path, model_name=YOLO_MODEL, margin=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load YOLO Pose Model
    print(f"Loading {model_name}...")
    yolo_model = YOLO(model_name)
    yolo_model.to(device)

    # 2. Load our LSTM Sequence Model
    print("Initializing LSTM Anomaly Model...")
    lstm_model = PoseActionLSTM().to(device)
    if os.path.exists("lstm_model.pth"):
        print("Loading trained weights from lstm_model.pth...")
        lstm_model.load_state_dict(torch.load("lstm_model.pth", map_location=device))
    else:
        print("Warning: lstm_model.pth not found. Using random weights!")
    lstm_model.eval()  # Set to inference mode

    # 3. Load Text Encoder (Sentence-Transformers)
    print("Loading Sentence-Transformers for Zero-Shot Classification...")
    text_encoder = SentenceTransformer('all-MiniLM-L6-v2').to(device)

    # Encode anchor texts representing Normal and Abnormal classes
    normal_text = "person walking normally, standing, or doing routine activities"
    abnormal_text = "person falling down, slipping, fainting, fighting, or moving abnormally"
    with torch.no_grad():
        anchor_embs = text_encoder.encode([normal_text, abnormal_text], convert_to_tensor=True)
        anchor_embs = torch.nn.functional.normalize(anchor_embs, p=2, dim=1)  # Shape: (2, 384)

    # Decision margin (calibrated on the validation set during training, if available)
    margin = resolve_margin(margin)

    # 4. Video IO setup
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

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    seq_len = SEQ_LEN
    # Per-tracked-person state
    history = {}            # id -> deque of keypoint feature vectors
    decision_history = {}   # id -> deque of recent raw decisions (for majority-vote smoothing)
    last_seen = {}          # id -> last frame index seen (for pruning)

    print(f"Starting video processing. Input: {input_path}, Output: {output_path}")
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO with tracking enabled to associate skeletons across frames
        results = yolo_model.track(frame, persist=True, verbose=False)

        annotated_frame = frame.copy()

        if results and len(results[0].boxes) > 0:
            result = results[0]
            # Automatically plot YOLO bounding boxes and skeletons
            annotated_frame = result.plot()

            # Extract keypoints
            if result.keypoints is not None and result.boxes.id is not None:
                keypoints = result.keypoints.xy.cpu().numpy()  # (num_people, 17, 2)
                kconf = result.keypoints.conf
                kconf = kconf.cpu().numpy() if kconf is not None else np.ones(keypoints.shape[:2], dtype=np.float32)
                boxes = result.boxes.xyxy.cpu().numpy()        # (num_people, 4)
                track_ids = result.boxes.id.int().cpu().numpy()

                for i, track_id in enumerate(track_ids):
                    if track_id not in history:
                        history[track_id] = deque(maxlen=seq_len)
                    last_seen[track_id] = frame_count

                    # Person-centric normalization + confidence masking (utils/pose.py).
                    # Must match prepare_data.py exactly so train/infer features align.
                    feat = normalize_keypoints(keypoints[i], kconf[i], boxes[i])
                    history[track_id].append(feat)

                    # If we have accumulated enough frames for this person, run LSTM
                    if len(history[track_id]) == seq_len:
                        seq_tensor = torch.tensor(np.array(history[track_id]), dtype=torch.float32).unsqueeze(0).to(device)

                        with torch.no_grad():
                            # Forward pass through LSTM (pose-only by default)
                            video_emb = lstm_model(seq_tensor)

                            # Zero-shot inference: cosine similarity with the text anchors.
                            # video_emb: (1, 384), anchor_embs: (2, 384)
                            sims = torch.mm(video_emb, anchor_embs.t())[0]
                            score = (sims[1] - sims[0]).item()

                        raw_abnormal = score > margin

                        # Majority-vote smoothing over recent frames to avoid flicker
                        if track_id not in decision_history:
                            decision_history[track_id] = deque(maxlen=SMOOTH_WINDOW)
                        decision_history[track_id].append(raw_abnormal)
                        votes = decision_history[track_id]
                        is_abnormal = sum(votes) > len(votes) / 2

                        # Overlay Anomaly Status
                        bbox = result.boxes.xyxy[i].cpu().numpy()
                        x1, y1, x2, y2 = map(int, bbox)

                        label = "Abnormal!" if is_abnormal else "Normal"
                        color = (0, 0, 255) if is_abnormal else (0, 255, 0)

                        # Draw label background and text
                        cv2.rectangle(annotated_frame, (x1, max(0, y1 - 40)), (x1 + 140, max(0, y1)), color, -1)
                        cv2.putText(annotated_frame, label, (x1 + 5, max(0, y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Bound memory on long videos: drop tracks that have disappeared.
        if frame_count % PRUNE_INTERVAL == 0:
            for tid in [t for t, fs in last_seen.items() if frame_count - fs > STALE_FRAMES]:
                history.pop(tid, None)
                decision_history.pop(tid, None)
                last_seen.pop(tid, None)

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
    parser = argparse.ArgumentParser(description="YOLO Pose + LSTM Anomaly Detection")
    parser.add_argument('--input', type=str, default='input.mp4', help='Path to input MP4 video')
    parser.add_argument('--output', type=str, default='output.mp4', help='Path to output MP4 video')
    parser.add_argument('--model', type=str, default=YOLO_MODEL, help='YOLO-Pose weights to use')
    parser.add_argument('--margin', type=float, default=None,
                        help='Decision margin (abnormal if sim_abnormal - sim_normal > margin). '
                             'Defaults to the calibrated value in threshold.json, else 0.0.')
    args = parser.parse_args()

    process_video(args.input, args.output, model_name=args.model, margin=args.margin)
