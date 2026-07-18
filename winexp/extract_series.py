"""Window-length experiment: extract a continuous 5 fps feature SERIES (T, 55)
from a video (single-worker), so windows of ANY length can be sliced offline
without re-running YOLO.

Usage: python winexp/extract_series.py <video_glob> <out_name>
"""
import glob
import os
import sys
from collections import deque  # noqa: F401  (parity with oneclass imports)

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "oneclass"))
from pose_features import (normalize_keypoints, sample_interval, resolve_yolo_weights,  # noqa: E402
                           pick_worker, GAP_RESET_SECONDS)
from ultralytics import YOLO  # noqa: E402

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def extract_series(video_path, yolo):
    cap = cv2.VideoCapture(video_path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    se = sample_interval(fps)
    gap_frames = int(GAP_RESET_SECONDS * fps)

    rows, gaps = [], []
    prev_center = None
    last_seen = None
    fi = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        fi += 1
        if fi % 600 == 0:
            print(f"  {fi}/{total}", flush=True)
        r = yolo.track(frame, persist=True, verbose=False)
        if fi % se != 0:
            continue
        if not (r and len(r[0].boxes) > 0) or r[0].keypoints is None:
            continue
        res = r[0]
        boxes = res.boxes.xyxy.cpu().numpy()
        if last_seen is not None and fi - last_seen > gap_frames:
            gaps.append(len(rows))          # series index where continuity broke
            prev_center = None
        wi, centers = pick_worker(boxes, prev_center)
        prev_center = centers[wi]
        last_seen = fi
        kp = res.keypoints.xy.cpu().numpy()
        kc = res.keypoints.conf
        kc = kc.cpu().numpy() if kc is not None else np.ones(kp.shape[:2], dtype=np.float32)
        rows.append(normalize_keypoints(kp[wi], kc[wi], boxes[wi], (W, H)))
    cap.release()
    return np.array(rows, dtype=np.float32), gaps


if __name__ == "__main__":
    video = glob.glob(sys.argv[1])[0]
    out = sys.argv[2]
    print("video:", video)
    yolo = YOLO(resolve_yolo_weights())
    S, gaps = extract_series(video, yolo)
    np.savez(os.path.join(OUT_DIR, out + ".npz"), series=S, gaps=np.array(gaps, dtype=np.int64))
    print(f"saved {out}.npz  T={len(S)} (~{len(S)/5:.0f}s)  gaps={gaps}")
