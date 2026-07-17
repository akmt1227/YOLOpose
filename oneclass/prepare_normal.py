"""Extract pose windows from NORMAL (correct work) videos only.

Single-worker mode: this station has ONE worker, so instead of relying on tracker
IDs (which fragment on large pose changes — e.g. bending down — and would reset the
13 s window exactly when something interesting happens), each sampled frame picks
the person nearest to the worker's previous position and appends to ONE continuous
history. The history only resets after the worker has been absent for a while.

Output (into --work_dir, default: this folder):
  X_normal.npy       (N, SEQ_LEN, POSE_DIM) float32 windows
  groups_normal.npy  (N,) source-video id per window (for leak-free calibration split)
"""
import os
import glob
import argparse
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

from pose_features import (normalize_keypoints, sample_interval, resolve_yolo_weights,
                           pick_worker, SEQ_LEN, STRIDE, GAP_RESET_SECONDS)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)


def extract_windows(video_path, yolo_model, seq_len=SEQ_LEN, stride=STRIDE):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  WARNING: cannot open {video_path}, skipping.")
        return []
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if width == 0 or height == 0:
        print(f"  WARNING: invalid dimensions for {video_path}, skipping.")
        cap.release()
        return []
    sample_every = sample_interval(fps)
    gap_reset_frames = int(GAP_RESET_SECONDS * fps)

    history = deque(maxlen=seq_len)   # ONE continuous history for THE worker
    count = 0
    prev_center = None
    last_seen_frame = None
    windows = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % 300 == 0:
            print(f"  Progress: {frame_idx}/{total} frames ({(frame_idx/max(1, total))*100:.1f}%)", flush=True)

        # Track every frame (stabler boxes); IDs themselves are not used.
        results = yolo_model.track(frame, persist=True, verbose=False)
        if frame_idx % sample_every != 0:
            continue
        if not (results and len(results[0].boxes) > 0) or results[0].keypoints is None:
            continue

        result = results[0]
        keypoints = result.keypoints.xy.cpu().numpy()
        kconf = result.keypoints.conf
        kconf = kconf.cpu().numpy() if kconf is not None else np.ones(keypoints.shape[:2], dtype=np.float32)
        boxes = result.boxes.xyxy.cpu().numpy()

        # Long absence -> the window is no longer continuous; start over.
        if last_seen_frame is not None and frame_idx - last_seen_frame > gap_reset_frames:
            history.clear()
            count = 0
            prev_center = None

        i, centers = pick_worker(boxes, prev_center)
        prev_center = centers[i]
        last_seen_frame = frame_idx

        history.append(normalize_keypoints(keypoints[i], kconf[i], boxes[i], (width, height)))
        count += 1
        if len(history) == seq_len and (count - seq_len) % stride == 0:
            windows.append(np.array(history, dtype=np.float32))

    cap.release()
    return windows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--normal_dir', type=str,
                        default=os.path.join(REPO_ROOT, 'dataset', 'normal'),
                        help='Folder with correct-work videos (normal only!)')
    parser.add_argument('--model', type=str, default=resolve_yolo_weights(),
                        help='YOLO-Pose weights (default: yolo26x-pose.pt, auto-downloaded)')
    parser.add_argument('--work_dir', type=str, default=SCRIPT_DIR,
                        help='Where to write X_normal.npy / groups_normal.npy')
    args = parser.parse_args()

    videos = sorted(glob.glob(os.path.join(args.normal_dir, '*.mp4')))
    if not videos:
        print(f"No .mp4 found in {args.normal_dir}")
        return

    print(f"Loading YOLO model ({args.model})...")
    yolo_model = YOLO(args.model)

    X, groups = [], []
    for vid, video in enumerate(videos):
        print(f"Processing {video}...")
        ws = extract_windows(video, yolo_model)
        print(f"  Extracted {len(ws)} windows (~13s each).")
        X.extend(ws)
        groups.extend([vid] * len(ws))

    if not X:
        print("No windows extracted. Videos must contain a trackable person for at least ~13s.")
        return

    X = np.array(X, dtype=np.float32)
    groups = np.array(groups, dtype=np.int64)
    np.save(os.path.join(args.work_dir, 'X_normal.npy'), X)
    np.save(os.path.join(args.work_dir, 'groups_normal.npy'), groups)
    print(f"Saved {len(X)} windows from {len(videos)} video(s)  shape={X.shape}")
    print(f"  -> {os.path.join(args.work_dir, 'X_normal.npy')}")


if __name__ == '__main__':
    main()
