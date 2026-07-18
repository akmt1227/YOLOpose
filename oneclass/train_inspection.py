"""Train the gate-5 inspection-phase model (OK-only): k-means over normal pose
frames, saving centroids + the calibrated inspection-cluster set.

The cluster SET below was calibrated once against operator ground truth
(normal0 inspection windows 10-18/26-32/39-45/52-60 s; skipped-video omissions
at 2-4 & 9-11 s with inspections resuming at ~17 s). The k-means fit itself
uses ONLY normal videos and is deterministic (fixed seed), so the indices
remain valid across retrains on the same data. If the normal dataset changes,
re-verify the set with scripts from the phase-discovery analysis.

Output: oneclass/inspection_phase.npz  (centroids, sel, params)
"""
import os
import sys

import numpy as np
from sklearn.cluster import KMeans

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from pose_features import STRIDE  # noqa: E402

K = 16
SEED = 42
SEL = [5, 6, 11, 15]     # inspection-phase clusters (see header)
# Additional slow-manipulation indicator: cluster C14 counts as inspection ONLY
# when local motion is low (careful cable turning ~0.012-0.019 vs brisk bench
# work ~0.031-0.042; threshold 0.024 sits in the gap). This recovered the
# no-pointing video's inspections (its style lands in C14) without breaking the
# skipped-video omission detection.
C_SLOW = 14
MOTION_MAX = 0.024

X = np.load(os.path.join(SCRIPT_DIR, "X_normal.npy"))
g = np.load(os.path.join(SCRIPT_DIR, "groups_normal.npy"))
series = []
for vid in np.unique(g):
    idx = np.where(g == vid)[0]
    s = [X[idx[0]]]
    for i in idx[1:]:
        s.append(X[i][-STRIDE:])
    series.append(np.vstack(s))
train = np.vstack(series)

km = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit(train)
out = os.path.join(SCRIPT_DIR, "inspection_phase.npz")
np.savez(out, centroids=km.cluster_centers_.astype(np.float32),
         sel=np.array(SEL, dtype=np.int64),
         c_slow=np.int64(C_SLOW), motion_max=np.float64(MOTION_MAX))
occ = np.bincount(km.predict(train), minlength=K) / len(train)
print(f"trained k-means on {len(train)} normal frames -> {out}")
print(f"inspection clusters {SEL} occupy {occ[SEL].sum()*100:.1f}% of normal time")
