import os
import cv2
import glob
import numpy as np
from collections import deque
from ultralytics import YOLO
from utils.pose import normalize_keypoints, SEQ_LEN, STRIDE
from utils.config import YOLO_MODEL, STALE_FRAMES, PRUNE_INTERVAL


def extract_keypoints(video_path, yolo_model, seq_len=SEQ_LEN, stride=STRIDE):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  WARNING: cannot open {video_path}, skipping.")
        return []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    history = {}    # track_id -> deque of per-frame feature vectors
    counts = {}     # track_id -> frames appended so far (for strided sampling)
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

        # track to keep IDs consistent
        results = yolo_model.track(frame, persist=True, verbose=False)
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

            # Person-centric normalization + confidence masking (utils/pose.py)
            history[track_id].append(normalize_keypoints(keypoints[i], kconf[i], boxes[i]))
            counts[track_id] += 1
            last_seen[track_id] = frame_idx

            # Emit a window every `stride` frames once full, instead of every frame.
            # This avoids ~29/30 overlap between consecutive training samples.
            if len(history[track_id]) == seq_len and (counts[track_id] - seq_len) % stride == 0:
                sequences.append(np.array(history[track_id], dtype=np.float32))

    cap.release()
    return sequences


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--normal_dir', type=str, default='dataset/normal', help='Path to normal videos')
    parser.add_argument('--abnormal_dir', type=str, default='dataset/abnormal', help='Path to abnormal videos')
    parser.add_argument('--model', type=str, default=YOLO_MODEL, help='YOLO-Pose weights to use')
    args = parser.parse_args()

    print(f"Loading YOLO model ({args.model})...")
    yolo_model = YOLO(args.model)

    classes = ["normal", "abnormal"]
    dataset_dirs = [args.normal_dir, args.abnormal_dir]

    X = []
    y = []
    groups = []       # source-video id per sequence -> enables leak-free (video-level) split
    video_id = 0

    for cls_idx, (cls_name, folder_path) in enumerate(zip(classes, dataset_dirs)):
        video_files = sorted(glob.glob(os.path.join(folder_path, "*.mp4")))

        for video_file in video_files:
            print(f"Processing {video_file}...")
            seqs = extract_keypoints(video_file, yolo_model)
            print(f"  Extracted {len(seqs)} sequences of {SEQ_LEN} frames.")

            X.extend(seqs)
            y.extend([cls_idx] * len(seqs))
            groups.extend([video_id] * len(seqs))
            video_id += 1

    if len(X) == 0:
        print("No data extracted. Please make sure .mp4 files are in dataset/normal and dataset/abnormal")
        return

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    groups = np.array(groups, dtype=np.int64)

    print(f"Total sequences extracted: {len(X)}  shape={X.shape}")
    print(f"Normal: {int(np.sum(y == 0))}, Abnormal: {int(np.sum(y == 1))}, Videos: {video_id}")

    np.save("X_data.npy", X)
    np.save("y_labels.npy", y)
    np.save("groups.npy", groups)
    print("Saved extracted data to X_data.npy, y_labels.npy and groups.npy")


if __name__ == "__main__":
    main()
