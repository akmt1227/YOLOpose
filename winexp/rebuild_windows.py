"""Rebuild oneclass/X_normal.npy at the CURRENT pose_features.SEQ_LEN by
re-slicing the continuous 5 fps series reconstructed from the existing windows
(identical features; no YOLO pass needed).

Run AFTER changing SEQ_LEN in oneclass/pose_features.py (e.g., 65 -> 30 for the
adopted 6 s window). Overwrites X_normal.npy / groups_normal.npy in oneclass/.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OC = os.path.join(ROOT, "oneclass")
sys.path.insert(0, OC)
from pose_features import SEQ_LEN, STRIDE  # noqa: E402  (the NEW values)

X = np.load(os.path.join(OC, "X_normal.npy"))
g = np.load(os.path.join(OC, "groups_normal.npy"))
OLD_L = X.shape[1]
OLD_STRIDE = 15   # the windows on disk were cut with stride 15

new_X, new_g = [], []
for vid in np.unique(g):
    idx = np.where(g == vid)[0]
    s = [X[idx[0]]]
    for i in idx[1:]:
        s.append(X[i][-OLD_STRIDE:])
    S = np.vstack(s)
    for i in range(0, len(S) - SEQ_LEN + 1, STRIDE):
        new_X.append(S[i:i + SEQ_LEN])
        new_g.append(vid)

new_X = np.array(new_X, dtype=np.float32)
new_g = np.array(new_g, dtype=np.int64)
np.save(os.path.join(OC, "X_normal.npy"), new_X)
np.save(os.path.join(OC, "groups_normal.npy"), new_g)
print(f"rebuilt: {OLD_L}-sample windows -> {SEQ_LEN}-sample windows")
print(f"X_normal {new_X.shape}  per-video {np.bincount(new_g).tolist()}")
