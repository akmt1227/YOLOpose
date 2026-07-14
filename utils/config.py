"""Central configuration shared across the YOLOpose pipeline."""

# YOLO-Pose weights used for extraction (prepare_data.py) and inference (main.py).
# Single source of truth. yolo26x-pose is the largest/most accurate pose model
# (slower, ~120MB); switch to a lighter one (e.g. yolov8m-pose.pt) here or via
# --model on the CLI if throughput matters more than accuracy.
YOLO_MODEL = "yolo26x-pose.pt"

# Memory bounding for long videos: drop a tracked person that has not been seen
# for STALE_FRAMES frames, sweeping every PRUNE_INTERVAL frames. Without this the
# per-track history dicts grow unbounded over a long clip.
STALE_FRAMES = 90       # ~3x the 30-frame sequence window
PRUNE_INTERVAL = 100

# Inference: majority-vote smoothing window (frames) for the anomaly label, so it
# does not flicker on a single noisy window.
SMOOTH_WINDOW = 10
