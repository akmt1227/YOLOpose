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

# --- Gate-4 gesture signature (EXPERIMENTAL — a cycle-rhythm proxy) -----------
# Intended target: the once-per-cycle pointing check (done with the RIGHT hand;
# this camera views the worker from behind/above, so YOLO's left/right labels
# are mirrored — k=9 "left wrist" is the worker's right hand).
#
# REALITY CHECK (frame-level analysis with operator ground truth): during the
# actual pointing the hand is largely occluded from this camera — wrist
# confidence collapses to 0.1–0.3 — so the pointing itself cannot be detected
# reliably (strict conf gate misses ~90% of real pointings in normal videos;
# a loose gate admits hallucinated positions). What this signature actually
# tracks is a nearby cycle-periodic arm extension toward the upper-right (tray
# direction). Its disruption correlated with the omitted pointings in the
# tested NG video, so the watchdog still fires usefully, BUT:
#   - the alert may clear EARLY when the proxy motion resumes before the real
#     pointing does (observed: cleared ~23.8 s, real first pointing ~33 s), and
#   - an omission that leaves the proxy motion intact will be missed.
# Treat gate 4 as EXPERIMENTAL. The robust path for this NG type is the
# supervised 5-class classifier in the main pipeline (whole-body, 13 s window).
# Parameters grid-searched on the 3 normal videos + the NG video.
POINTING_EXT_MIN = 0.32               # wrist distance from body center
POINTING_X_MIN = 0.10                 # extended to the right
POINTING_Y_MAX = -0.15                # and upward (image y grows downward)
POINTING_SUSTAIN = 1                  # samples the pose must hold to count as an event

# STRICT (real-pointing) signature: forward-DOWN wrist on RAW keypoints at full
# framerate. Validated on the NG video: fires only at the true pointings
# (33.6 s, 46.2 s per operator ground truth) and never during the omission
# span. On normal videos the pointing hand is too often occluded for this to
# be the primary signal, so detect.py uses it asymmetrically: only to CLEAR an
# active overdue alert (a single proxy event no longer clears; two separate
# proxy events remain as a fallback clear).
POINTING_STRICT_CONF = 0.3
POINTING_STRICT_X_MIN = 0.10
POINTING_STRICT_Y_MIN = 0.02          # BELOW body center (pointing at the sheet)
POINTING_STRICT_SUSTAIN_SECONDS = 0.15
OVERDUE_PROXY_CLEAR_COUNT = 2
POINTING_TIMEOUT_SECONDS = 20.0       # gap between pointings (normal max gap 16.4 s)
# Before the FIRST pointing after observation starts (or after the worker was
# absent), a tighter deadline applies: across the normal videos the first
# pointing always appeared within 12.8 s, so 16 s still has margin and alerts
# ~4 s sooner on an omitted first cycle.
POINTING_FIRST_DEADLINE_SECONDS = 16.0

# Note on left/right: this footage views the worker from behind/above, so
# YOLO's anatomical left/right labels are mirrored — keypoint k=9 ("left
# wrist", dims 27/28/29) is actually the worker's RIGHT hand, the one that
# performs the pointing check (confirmed with the operator).


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
