"""Pose features for the one-class prototype.

Deliberately a self-contained copy of the feature spec in ../utils/pose.py, so this
prototype stays fully independent of the supervised pipeline: changes here never
affect it, and vice versa. Keep the constants in sync manually if you want results
to stay comparable.

Feature per sampled frame = 55 dims:
  - 17 keypoints x (x, y, confidence) = 51, person-centric (bbox-normalized),
    low-confidence joints zeroed.
  - 4 absolute bbox features (cx/W, cy/H, w/W, h/H) — fixed-camera assumption.

A window = SEQ_LEN samples at ~TARGET_FPS = 65 / 5 = 13 s (one inspection cycle).
"""
import os

import numpy as np

# YOLO-Pose weights for the prototype (single source of truth).
# yolo26x = largest/most accurate variant; auto-downloaded by Ultralytics on first
# use. Heavy — expect slow processing without a CUDA GPU.
YOLO_WEIGHTS = "yolo26x-pose.pt"


def resolve_yolo_weights(name=YOLO_WEIGHTS):
    """Prefer a local copy at the repo root; otherwise return the bare name so
    Ultralytics auto-downloads it (into the current working directory)."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local = os.path.join(repo_root, name)
    return local if os.path.exists(local) else name

NUM_KEYPOINTS = 17
KP_DIM = NUM_KEYPOINTS * 3            # (x, y, conf) per keypoint -> 51
BOX_DIM = 4                           # absolute bbox cx, cy, w, h (image-normalized)
POSE_DIM = KP_DIM + BOX_DIM           # 55

TARGET_FPS = 5                        # feature sampling rate
SEQ_LEN = 65                          # samples per window (65 / 5 fps = 13 s)
STRIDE = 15                           # step (in samples) between windows (~3 s)

KEYPOINT_CONF_THRESHOLD = 0.3         # joints below this confidence are treated as missing

GAP_RESET_SECONDS = 2.0               # worker unseen for this long -> break window continuity

# --- Pointing-check gesture signature ---
# Once per ~13 s cycle the worker extends the LEFT arm up-and-right (toward the
# inspection board). A differential test on the "no pointing" NG video showed
# exactly this event missing in the offending cycles, so its absence is the NG
# signal. Parameters grid-searched on the 3 normal videos + the NG video:
# normal max event gap = 16.0 s, first NG omission gap = 23.8 s -> timeout 20 s
# gives a ~4 s margin both ways. (The NG video's 2nd omission was cut short by
# the video ending; in continuous operation it would also hit the timeout.)
POINTING_EXT_MIN = 0.32               # wrist distance from body center
POINTING_X_MIN = 0.10                 # extended to the right
POINTING_Y_MAX = -0.15                # and upward (image y grows downward)
POINTING_SUSTAIN = 1                  # samples the pose must hold to count as an event
POINTING_TIMEOUT_SECONDS = 20.0       # no pointing for ~1.5 cycles -> NG


def is_pointing_pose(feat):
    """Does this 55-dim pose feature match the pointing gesture? (left wrist, k=9)"""
    lx, ly, lc = float(feat[27]), float(feat[28]), float(feat[29])
    if lc <= KEYPOINT_CONF_THRESHOLD:
        return False
    ext = (lx * lx + ly * ly) ** 0.5
    return ext > POINTING_EXT_MIN and lx > POINTING_X_MIN and ly < POINTING_Y_MAX


def pick_worker(boxes, prev_center):
    """Single-worker station: index of THE worker among detected people.

    Tracker IDs fragment on large pose changes (bending to pick up a dropped part)
    and would reset the 13 s window exactly when something interesting happens, so
    we do not use them: pick the person nearest to the worker's previous position
    (or the largest box when there is no history) and keep ONE continuous history.

    Returns (index, centers) where centers is (num_people, 2).
    """
    boxes = np.asarray(boxes, dtype=np.float32)
    centers = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], axis=1)
    if prev_center is None:
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        return int(areas.argmax()), centers
    d = np.linalg.norm(centers - np.asarray(prev_center, dtype=np.float32), axis=1)
    return int(d.argmin()), centers


def sample_interval(fps):
    """How many source frames to skip between feature samples for this video."""
    if not fps or fps <= 0:
        fps = 30.0
    return max(1, round(fps / TARGET_FPS))


def normalize_keypoints(xy, conf, box, frame_size):
    """Pose features for one person in one frame -> (55,) float32."""
    xy = np.asarray(xy, dtype=np.float32).reshape(NUM_KEYPOINTS, 2).copy()
    conf = np.asarray(conf, dtype=np.float32).reshape(NUM_KEYPOINTS)

    x1, y1, x2, y2 = [float(v) for v in box]
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    scale = max(x2 - x1, y2 - y1) + 1e-6

    xy[:, 0] = (xy[:, 0] - cx) / scale     # center on the person, not the image
    xy[:, 1] = (xy[:, 1] - cy) / scale
    xy[conf < KEYPOINT_CONF_THRESHOLD] = 0.0

    kp_feat = np.concatenate([xy, conf[:, None]], axis=1).reshape(-1)   # (51,)

    W, H = float(frame_size[0]) + 1e-6, float(frame_size[1]) + 1e-6
    box_feat = np.array([cx / W, cy / H, (x2 - x1) / W, (y2 - y1) / H], dtype=np.float32)
    return np.concatenate([kp_feat, box_feat])                          # (55,)


def motion_energy(window):
    """Mean per-step displacement of the person-centric keypoints over a window.

    Uses only the person-centric xy dims, so it measures how much the LIMBS move —
    a worker standing in place but moving their hands still scores well above zero.
    Near-zero energy across a whole window = idle / not working. Used by the rule
    gate in detect.py because autoencoders reconstruct degenerate (static) inputs
    deceptively well, so "frozen worker" needs an explicit check.

    window: (S, POSE_DIM) -> float
    """
    window = np.asarray(window, dtype=np.float32)
    kp = window[:, :KP_DIM].reshape(len(window), NUM_KEYPOINTS, 3)[:, :, :2]
    if len(kp) < 2:
        return 0.0
    return float(np.abs(np.diff(kp, axis=0)).mean())
