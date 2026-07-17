"""One-class NG detection on a video: alert on anything that deviates from normal work.

Single-worker mode: one continuous pose history follows THE worker (nearest person
to the previous position), so tracker-ID switches — which happen exactly during
dramatic motions like bending to pick up a dropped part — no longer reset the 13 s
window and blind the detector.

Three gates, any of which raises the red NG banner:
  1. Reconstruction gate — autoencoder error above the calibrated threshold
     -> "DEVIATES FROM NORMAL WORK" (unknown/unspecified deviation)
  2. Idle gate — worker present but limb motion energy near zero for a window
     -> "WORKER IDLE"  (autoencoders rebuild static input too well, hence a rule)
  3. Absence gate — no person detected for ABSENCE_SECONDS
     -> "NO WORKER PRESENT"
"""
import os
import json
import argparse
import warnings
from collections import deque, Counter

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from model import PoseLSTMAutoencoder, window_scores
from pose_features import (normalize_keypoints, motion_energy, sample_interval,
                           resolve_yolo_weights, pick_worker, is_pointing_pose,
                           SEQ_LEN, GAP_RESET_SECONDS, TARGET_FPS,
                           POINTING_SUSTAIN, POINTING_TIMEOUT_SECONDS,
                           POINTING_FIRST_DEADLINE_SECONDS)

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

SMOOTH_WINDOW = 10        # majority vote over recent per-window verdicts (~2 s)
NG_HOLD_FRAMES = 90       # keep the banner visible ~3 s
ABSENCE_SECONDS = 10.0    # no worker for this long -> NG

# Anomaly-recency gate: a brief event stays inside the sliding 13 s window (and
# keeps the score high) for 13 s, which would leave NG on screen long after the
# event. Only raise NG while the PEAK per-step error sits in the most recent part
# of the window, so the alert hugs the event itself. Exception: right after the
# window first fills (cold start), its whole content is still unreported, so any
# peak counts — this is what lets a drop in the first seconds of a clip be caught.
RECENT_SECONDS = 4.0

REASONS = {
    'dev':         "NG: DEVIATES FROM NORMAL WORK",
    'idle':        "NG: WORKER IDLE",
    'absent':      "NG: NO WORKER PRESENT",
    'no_pointing': "NG: NO POINTING CHECK",
}


def draw_banner(frame, text):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 255), -1)
    cv2.putText(frame, text, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)


def draw_score(frame, score, threshold):
    """Live anomaly score vs threshold, top-right (useful while tuning the prototype)."""
    w = frame.shape[1]
    txt = f"score {score:.4f} / thr {threshold:.4f}"
    cv2.putText(frame, txt, (w - 420, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


def process_video(input_path, output_path, work_dir=SCRIPT_DIR, yolo_weights=None,
                  recon_threshold=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load calibration
    thr_path = os.path.join(work_dir, 'threshold.json')
    if not os.path.exists(thr_path):
        print(f"{thr_path} not found. Run prepare_normal.py then train_ae.py first.")
        return
    with open(thr_path) as f:
        cal = json.load(f)
    recon_thr = recon_threshold if recon_threshold is not None else float(cal['recon_threshold'])
    idle_thr = float(cal['idle_threshold'])
    print(f"Thresholds: recon > {recon_thr:.5f} -> deviate | motion < {idle_thr:.6f} -> idle")

    # Load models
    ae = PoseLSTMAutoencoder().to(device)
    ae_path = os.path.join(work_dir, 'ae_model.pth')
    if not os.path.exists(ae_path):
        print(f"{ae_path} not found. Run train_ae.py first.")
        return
    ae.load_state_dict(torch.load(ae_path, map_location=device))
    ae.eval()

    yolo_weights = yolo_weights or resolve_yolo_weights()
    print(f"Loading {yolo_weights}...")
    yolo_model = YOLO(yolo_weights)
    yolo_model.to(device)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {input_path}")
        return
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    if width == 0 or height == 0:
        print(f"Error: Invalid video dimensions {width}x{height}")
        cap.release()
        return
    sample_every = sample_interval(fps)
    absence_frames = int(ABSENCE_SECONDS * fps)
    gap_reset_frames = int(GAP_RESET_SECONDS * fps)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Single-worker state (one continuous history; no tracker IDs)
    history = deque(maxlen=SEQ_LEN)
    verdicts = deque(maxlen=SMOOTH_WINDOW)   # 'ok' / 'dev' / 'idle'
    recent_samples = int(RECENT_SECONDS * TARGET_FPS)
    decisions_since_full = 0
    prev_center = None
    last_sample_frame = None

    # Gate 4: pointing watchdog — the worker must show the pointing gesture at
    # least once every POINTING_TIMEOUT_SECONDS; before the FIRST pointing after
    # observation starts, the tighter POINTING_FIRST_DEADLINE_SECONDS applies.
    last_pointing_time = None   # seconds; None until the worker is first sampled
    pointing_seen = False       # any pointing observed since observation/gap reset?
    pointing_run = 0
    pointing_logged = False
    is_ng, reason = False, None
    last_score = None
    last_person_frame = 0
    ng_hold, banner_text = 0, "NG"
    absent_logged = False
    frame_count = 0

    print(f"Starting one-class detection. Input: {input_path}, Output: {output_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model.track(frame, persist=True, verbose=False)
        sampled = (frame_count % sample_every == 0)
        annotated = frame.copy()
        ng_reason_now = None
        chip_box = None   # worker bbox; the chip itself is drawn after the banner state is known

        has_person = bool(results and len(results[0].boxes) > 0
                          and results[0].keypoints is not None)
        if has_person:
            result = results[0]
            # labels=False: suppress YOLO's own "person 0.9" text, which overlapped
            # our OK/NG chip drawn at the same spot above the bbox.
            annotated = result.plot(labels=False, conf=False)
            last_person_frame = frame_count
            absent_logged = False

            boxes = result.boxes.xyxy.cpu().numpy()
            wi, centers = pick_worker(boxes, prev_center)
            prev_center = centers[wi]

            if sampled:
                # Long absence broke continuity -> restart the window.
                now_s = frame_count / fps
                if last_sample_frame is not None and frame_count - last_sample_frame > gap_reset_frames:
                    history.clear()
                    verdicts.clear()
                    decisions_since_full = 0
                    last_pointing_time = now_s   # absence gap -> restart the grace period
                    pointing_seen = False
                    pointing_run = 0
                last_sample_frame = frame_count

                keypoints = result.keypoints.xy.cpu().numpy()
                kconf = result.keypoints.conf
                kconf = kconf.cpu().numpy() if kconf is not None else np.ones(keypoints.shape[:2], dtype=np.float32)
                feat = normalize_keypoints(keypoints[wi], kconf[wi], boxes[wi], (width, height))
                history.append(feat)

                # Gate 4 bookkeeping: note every observed pointing gesture.
                if last_pointing_time is None:
                    last_pointing_time = now_s   # grace period starts at first observation
                if is_pointing_pose(feat):
                    pointing_run += 1
                    if pointing_run >= POINTING_SUSTAIN:
                        last_pointing_time = now_s
                        pointing_seen = True
                        pointing_logged = False
                else:
                    pointing_run = 0

                if len(history) == SEQ_LEN:
                    window = np.array(history, dtype=np.float32)

                    # Gate 2: idle (checked first — static input also fools the AE)
                    if motion_energy(window) < idle_thr:
                        verdict = 'idle'
                        last_score = None
                    else:
                        # Gate 1: reconstruction error + anomaly-recency gate
                        x = torch.tensor(window).unsqueeze(0).to(device)
                        score, err = window_scores(ae, x)
                        last_score = float(score[0])
                        verdict = 'ok'
                        if last_score > recon_thr:
                            peak = int(err[0].argmax().item())
                            is_recent = peak >= SEQ_LEN - recent_samples
                            cold_start = decisions_since_full < recent_samples
                            if is_recent or cold_start:
                                verdict = 'dev'
                    decisions_since_full += 1
                    verdicts.append(verdict)

                    ng_votes = [v for v in verdicts if v != 'ok']
                    is_ng = len(ng_votes) > len(verdicts) / 2
                    reason = Counter(ng_votes).most_common(1)[0][0] if is_ng else None

            # Remember where the worker chip goes; it is drawn AFTER the banner
            # state is decided so chip and banner always agree.
            if len(verdicts) > 0:
                if is_ng and reason:
                    ng_reason_now = reason
                chip_box = tuple(map(int, boxes[wi]))

        # Gate 4: pointing watchdog (overrides dev/idle: more specific reason).
        # Tighter deadline before the first pointing is ever seen (cold start).
        pointing_elapsed = None
        if last_pointing_time is not None:
            deadline = POINTING_TIMEOUT_SECONDS if pointing_seen else POINTING_FIRST_DEADLINE_SECONDS
            elapsed = frame_count / fps - last_pointing_time
            if elapsed > deadline:
                pointing_elapsed = elapsed
                ng_reason_now = 'no_pointing'
                if not pointing_logged:
                    print(f"  [NG] no_pointing: none seen since ~{last_pointing_time:.1f}s "
                          f"(now {frame_count/fps:.1f}s, deadline {deadline:.0f}s)", flush=True)
                    pointing_logged = True

        # Gate 3: absence watchdog
        if frame_count - last_person_frame > absence_frames:
            ng_reason_now = 'absent'
            if not absent_logged:
                print(f"  [NG] absent since ~{last_person_frame/fps:.1f}s "
                      f"(now {frame_count/fps:.1f}s)", flush=True)
                absent_logged = True

        # Banner
        if ng_reason_now is not None:
            banner_text = REASONS.get(ng_reason_now, "NG")
            if ng_reason_now == 'no_pointing' and pointing_elapsed is not None:
                # Show the live elapsed time: "20s without a pointing check" is
                # self-explanatory, so the timeout-style timing reads correctly.
                banner_text = f"{banner_text} ({pointing_elapsed:.0f}s)"
            if ng_hold == 0 and ng_reason_now not in ('absent', 'no_pointing'):
                print(f"  [NG] {ng_reason_now} at frame {frame_count} (~{frame_count/fps:.1f}s)",
                      flush=True)
            ng_hold = NG_HOLD_FRAMES
        banner_visible = ng_hold > 0
        if banner_visible:
            draw_banner(annotated, banner_text)
            ng_hold -= 1

        # Worker chip, ALIGNED with the banner: whenever the red banner is up
        # (any gate: dev/idle/no_pointing), the chip also reads NG. Operators
        # were confused by "OK" on the worker while an NG banner was showing.
        if chip_box is not None:
            x1, y1, x2, y2 = chip_box
            chip_ng = banner_visible
            label = "NG" if chip_ng else "OK"
            color = (0, 0, 255) if chip_ng else (0, 255, 0)
            cv2.rectangle(annotated, (x1, max(0, y1 - 40)), (x1 + 80, max(0, y1)), color, -1)
            cv2.putText(annotated, label, (x1 + 5, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if last_score is not None:
            draw_score(annotated, last_score, recon_thr)

        out.write(annotated)
        frame_count += 1
        if frame_count % 300 == 0:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"  Progress: {frame_count}/{total} frames "
                  f"({(frame_count/max(1, total))*100:.1f}%)", flush=True)

    cap.release()
    out.release()
    print(f"Finished! Video saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="One-class work-motion NG detection")
    parser.add_argument('--input', type=str, required=True, help='Input MP4 video')
    parser.add_argument('--output', type=str, default='result_oneclass.mp4', help='Output MP4')
    parser.add_argument('--work_dir', type=str, default=SCRIPT_DIR,
                        help='Folder holding ae_model.pth / threshold.json')
    parser.add_argument('--model', type=str, default=None, help='YOLO-Pose weights')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Override the calibrated reconstruction threshold')
    args = parser.parse_args()

    process_video(args.input, args.output, work_dir=args.work_dir,
                  yolo_weights=args.model, recon_threshold=args.threshold)
