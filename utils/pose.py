"""Shared pose feature utilities (used by prepare_data.py and main.py).

Feature per frame = 17 keypoints x (x, y, confidence) = 51 dims, where x/y are
normalized to be translation- and scale-invariant relative to each person's
bounding box, and low-confidence (occluded/missing) joints are zeroed so the
LSTM does not treat YOLO's (0,0) placeholders as real positions at the origin.
"""
import numpy as np

NUM_KEYPOINTS = 17
POSE_DIM = NUM_KEYPOINTS * 3          # (x, y, conf) per keypoint -> 51
SEQ_LEN = 30                          # frames per sequence window
STRIDE = 15                           # step between consecutive windows (reduces 29/30 overlap)
KEYPOINT_CONF_THRESHOLD = 0.3         # joints below this confidence are treated as missing


def normalize_keypoints(xy, conf, box):
    """Person-centric normalization of one person's keypoints for one frame.

    Args:
        xy:   (17, 2) keypoint pixel coords (from result.keypoints.xy).
        conf: (17,)   per-keypoint confidence (from result.keypoints.conf).
        box:  (4,)    person bounding box as xyxy.

    Returns:
        (51,) float32 feature vector: 17 * (x_norm, y_norm, conf).
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

    feat = np.concatenate([xy, conf[:, None]], axis=1)   # (17, 3)
    return feat.reshape(-1)                               # (51,)
