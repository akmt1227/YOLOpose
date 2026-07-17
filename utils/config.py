"""Central configuration shared across the YOLOpose pipeline."""

# YOLO-Pose weights used for extraction (prepare_data.py) and inference (main.py).
# Single source of truth. The repo also ships yolo26l-pose.pt (larger/slower);
# point this at it, or pass --model on the CLI, to switch.
YOLO_MODEL = "yolov8m-pose.pt"

# --- Classes ------------------------------------------------------------------
# Index 0 must stay 'normal'. NG classes map to dataset/abnormal/<slug>/ folders:
#   dataset/normal/                 OK (correct work)
#   dataset/abnormal/too_long/     外観の検査時間が長い
#   dataset/abnormal/no_pointing/  検査ボードの指差し確認がない
#   dataset/abnormal/drop/         物を落下させている
#   dataset/abnormal/skipped/      外観検査をしていない箇所がある
CLASSES = ["normal", "too_long", "no_pointing", "drop", "skipped"]

# Overlay text per class (ASCII only: cv2.putText cannot render Japanese).
CLASS_DISPLAY = {
    "normal": "OK",
    "too_long": "NG: INSPECTION TOO LONG",
    "no_pointing": "NG: NO POINTING CHECK",
    "drop": "NG: OBJECT DROPPED",
    "skipped": "NG: SKIPPED INSPECTION",
}

# Memory bounding for long videos: drop a tracked person that has not been seen
# for STALE_FRAMES frames, sweeping every PRUNE_INTERVAL frames. STALE is generous
# (~10s) so a brief occlusion mid-cycle does not discard the track.
STALE_FRAMES = 300
PRUNE_INTERVAL = 100

# Inference smoothing: majority-vote window (in per-window decisions, ~2 s) for the
# per-person NG label, so it does not flicker on a single noisy window.
SMOOTH_WINDOW = 10

# NG alert (on-screen): once any person is flagged NG, hold the full-frame red
# banner for this many frames (~3s at 30fps) so the alert is clearly visible.
NG_HOLD_FRAMES = 90
