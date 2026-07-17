import os
import cv2
import glob
import numpy as np
from collections import deque
from ultralytics import YOLO
from utils.pose import normalize_keypoints, sample_interval, SEQ_LEN, STRIDE
from utils.config import YOLO_MODEL, CLASSES, STALE_FRAMES, PRUNE_INTERVAL


def collect_videos(normal_dir, abnormal_dir):
    """List (video_path, class_idx) pairs.

    Layout: normal videos directly in normal_dir; each NG type in its own
    subfolder of abnormal_dir named after the class slug (utils/config.py:CLASSES).
    """
    items = [(f, 0) for f in sorted(glob.glob(os.path.join(normal_dir, "*.mp4")))]
    for cls_idx, slug in enumerate(CLASSES[1:], start=1):
        folder = os.path.join(abnormal_dir, slug)
        items += [(f, cls_idx) for f in sorted(glob.glob(os.path.join(folder, "*.mp4")))]

    stray = sorted(glob.glob(os.path.join(abnormal_dir, "*.mp4")))
    if stray:
        print(f"WARNING: {len(stray)} video(s) directly under {abnormal_dir} are IGNORED.")
        print(f"  Sort them into per-NG-type subfolders: {CLASSES[1:]}")
        for f in stray:
            print(f"  ignored: {f}")
    return items


def extract_keypoints(video_path, yolo_model, seq_len=SEQ_LEN, stride=STRIDE):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  WARNING: cannot open {video_path}, skipping.")
        return []
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if width == 0 or height == 0:
        print(f"  WARNING: invalid dimensions for {video_path}, skipping.")
        cap.release()
        return []
    sample_every = sample_interval(fps)   # source frames per feature sample (~5 fps)

    history = {}    # track_id -> deque of per-sample feature vectors
    counts = {}     # track_id -> samples appended so far (for strided sampling)
    last_seen = {}  # track_id -> last frame index seen (for pruning stale tracks)
    sequences = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"  Progress: {frame_idx}/{total_frames} frames ({(frame_idx/max(1, total_frames))*100:.1f}%)", flush=True)

        # Bound memory on long videos: drop tracks that have disappeared.
        if frame_idx % PRUNE_INTERVAL == 0:
            for tid in [t for t, fs in last_seen.items() if frame_idx - fs > STALE_FRAMES]:
                history.pop(tid, None)
                counts.pop(tid, None)
                last_seen.pop(tid, None)

        # Track every frame so IDs stay stable across the full cycle...
        results = yolo_model.track(frame, persist=True, verbose=False)

        # ...but only sample features at ~TARGET_FPS (temporal downsampling).
        if frame_idx % sample_every != 0:
            continue
        if not (results and len(results[0].boxes) > 0):
            continue

        result = results[0]
        if result.keypoints is None or result.boxes.id is None:
            continue

        keypoints = result.keypoints.xy.cpu().numpy()          # (num_people, 17, 2)
        kconf = result.keypoints.conf
        kconf = kconf.cpu().numpy() if kconf is not None else np.ones(keypoints.shape[:2], dtype=np.float32)
        boxes = result.boxes.xyxy.cpu().numpy()                 # (num_people, 4)
        track_ids = result.boxes.id.int().cpu().numpy()

        for i, track_id in enumerate(track_ids):
            if track_id not in history:
                history[track_id] = deque(maxlen=seq_len)
                counts[track_id] = 0

            # Person-centric normalization + conf masking + absolute bbox (utils/pose.py)
            history[track_id].append(
                normalize_keypoints(keypoints[i], kconf[i], boxes[i], (width, height)))
            counts[track_id] += 1
            last_seen[track_id] = frame_idx

            # Emit a window every `stride` samples once full, instead of every sample.
            if len(history[track_id]) == seq_len and (counts[track_id] - seq_len) % stride == 0:
                sequences.append(np.array(history[track_id], dtype=np.float32))

    cap.release()
    return sequences


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--normal_dir', type=str, default='dataset/normal', help='Correct-work videos')
    parser.add_argument('--abnormal_dir', type=str, default='dataset/abnormal',
                        help=f'NG videos, in per-type subfolders: {CLASSES[1:]}')
    parser.add_argument('--model', type=str, default=YOLO_MODEL, help='YOLO-Pose weights to use')
    args = parser.parse_args()

    items = collect_videos(args.normal_dir, args.abnormal_dir)
    if not items:
        print("No videos found. Expected layout:")
        print(f"  {args.normal_dir}/*.mp4")
        for slug in CLASSES[1:]:
            print(f"  {args.abnormal_dir}/{slug}/*.mp4")
        return

    print(f"Loading YOLO model ({args.model})...")
    yolo_model = YOLO(args.model)

    X, y, groups = [], [], []
    for video_id, (video_file, cls_idx) in enumerate(items):
        print(f"Processing [{CLASSES[cls_idx]}] {video_file}...")
        seqs = extract_keypoints(video_file, yolo_model)
        print(f"  Extracted {len(seqs)} sequences (~13s each).")
        X.extend(seqs)
        y.extend([cls_idx] * len(seqs))
        groups.extend([video_id] * len(seqs))

    if len(X) == 0:
        print("No sequences extracted. Videos must contain a trackable person for at least ~13s.")
        return

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    groups = np.array(groups, dtype=np.int64)

    print(f"Total sequences extracted: {len(X)}  shape={X.shape}")
    for cls_idx, name in enumerate(CLASSES):
        n_seq = int(np.sum(y == cls_idx))
        n_vid = len(set(groups[y == cls_idx].tolist()))
        print(f"  {name}: {n_seq} sequences from {n_vid} video(s)")

    np.save("X_data.npy", X)
    np.save("y_labels.npy", y)
    np.save("groups.npy", groups)
    print("Saved extracted data to X_data.npy, y_labels.npy and groups.npy")


if __name__ == "__main__":
    main()
