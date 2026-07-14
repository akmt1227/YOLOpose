"""Central configuration shared across the YOLOpose pipeline."""

# YOLO-Pose weights used for extraction (prepare_data.py) and inference (main.py).
# Single source of truth. The repo also ships yolo26l-pose.pt (larger/slower);
# point this at it, or pass --model on the CLI, to switch.
YOLO_MODEL = "yolov8m-pose.pt"

# Memory bounding for long videos: drop a tracked person that has not been seen
# for STALE_FRAMES frames, sweeping every PRUNE_INTERVAL frames. Without this the
# per-track history dicts grow unbounded over a long clip.
STALE_FRAMES = 90       # ~3x the 30-frame sequence window
PRUNE_INTERVAL = 100

# Inference: majority-vote smoothing window (frames) for the anomaly label, so it
# does not flicker on a single noisy window.
SMOOTH_WINDOW = 10
