"""Shared pose feature utilities (used by prepare_data.py and main.py).

Feature per sampled frame = 55 dims:
  - 17 keypoints x (x, y, confidence) = 51, person-centric (translation/scale
    invariant relative to the person's bounding box), low-confidence joints zeroed.
  - 4 absolute bbox features (cx/W, cy/H, w/W, h/H) — WHERE in the scene the person
    is working. Fixed-camera assumption: the inspection board / work area sit at
    fixed screen positions, so absolute position carries task information
    (pointing at the board, which area is being inspected).

Sequence sampling covers one full inspection cycle (~13 s):
  features are sampled at ~TARGET_FPS and a window holds SEQ_LEN samples,
  so SEQ_LEN / TARGET_FPS = 65 / 5 = 13 s regardless of the source video fps.
"""
import numpy as np

NUM_KEYPOINTS = 17
KP_DIM = NUM_KEYPOINTS * 3            # (x, y, conf) per keypoint -> 51
BOX_DIM = 4                           # absolute bbox cx, cy, w, h (image-normalized)
POSE_DIM = KP_DIM + BOX_DIM           # 55

# --- Sequence sampling (sized to a ~13 s inspection cycle) ---
TARGET_FPS = 5                        # feature sampling rate
SEQ_LEN = 65                          # samples per window (65 / 5 fps = 13 s)
STRIDE = 15                           # step (in samples) between windows (~3 s)

KEYPOINT_CONF_THRESHOLD = 0.3         # joints below this confidence are treated as missing

GAP_RESET_SECONDS = 2.0               # worker unseen for this long -> break window continuity


def pick_worker(boxes, prev_center):
    """Single-worker station: index of THE worker among detected people.

    Tracker IDs fragment on large pose changes (e.g. bending to pick up a
    dropped part) and would reset the 13 s window exactly when something
    interesting happens, so IDs are not used: pick the person nearest to the
    worker's previous position (largest box when there is no history) and keep
    ONE continuous history. Also keeps the neighboring station's worker out of
    the data. (Ported from the oneclass prototype where it was validated.)

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
    """Pose features for one person in one frame.

    Args:
        xy:   (17, 2) keypoint pixel coords (from result.keypoints.xy).
        conf: (17,)   per-keypoint confidence (from result.keypoints.conf).
        box:  (4,)    person bounding box as xyxy (pixels).
        frame_size: (width, height) of the video frame.

    Returns:
        (55,) float32: 17 * (x_norm, y_norm, conf) person-centric + absolute bbox.
    """
    xy = np.asarray(xy, dtype=np.float32).reshape(NUM_KEYPOINTS, 2).copy()
    conf = np.asarray(conf, dtype=np.float32).reshape(NUM_KEYPOINTS)

    x1, y1, x2, y2 = [float(v) for v in box]
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    scale = max(x2 - x1, y2 - y1) + 1e-6   # bbox size -> scale invariance

    xy[:, 0] = (xy[:, 0] - cx) / scale     # center on the person, not the image
    xy[:, 1] = (xy[:, 1] - cy) / scale

    # Zero out unreliable joints so their placeholder positions do not leak in.
    xy[conf < KEYPOINT_CONF_THRESHOLD] = 0.0

    kp_feat = np.concatenate([xy, conf[:, None]], axis=1).reshape(-1)   # (51,)

    # Absolute scene position (fixed-camera assumption).
    W, H = float(frame_size[0]) + 1e-6, float(frame_size[1]) + 1e-6
    box_feat = np.array([cx / W, cy / H, (x2 - x1) / W, (y2 - y1) / H], dtype=np.float32)

    return np.concatenate([kp_feat, box_feat])                          # (55,)
