"""Gates 5+6: inspection-phase watchdog + too-long episode metering (OK-only).

The visual inspection (worker turns from the bench, head down over the part
held in both hands, ~7 s per cycle) occupies a distinct region of pose-feature
space, discovered by clustering NORMAL videos only. Per sampled frame the
55-dim feature is matched to the nearest k-means centroid; membership in the
calibrated inspection-cluster set, sustained (>=SUSTAIN_MIN samples within
SUSTAIN_WINDOW seconds), counts as "inspection observed" and feeds a watchdog
timer.

Calibration (3 normal videos + operator ground truth on the skipped video):
  worst normal first event 8.6 s / max gap 13.2 s;
  skipped video: zero events during the omission span, resume at 17.4 s.
Deadlines chosen between those: first 13 s, steady 16 s. Margins are ~2.8 s —
re-verify when normal footage is added.
"""
import os
from collections import deque

import numpy as np

from pose_features import motion_energy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "inspection_phase.npz")

SUSTAIN_WINDOW = 2.0      # seconds
SUSTAIN_MIN = 3           # inspection-cluster samples within the window
INSPECTION_FIRST_DEADLINE_SECONDS = 13.0
INSPECTION_TIMEOUT_SECONDS = 16.0

# --- Gate 6 (EXPERIMENTAL): inspection running too long ----------------------
# A deliberately stretched inspection shows as SUSTAINED slow manipulation
# (C14 + low motion): runtime-measured 10.2 s on the too_long video's second
# event vs <= 6.6 s across every normal-style inspection (runtime matches the
# offline calibration). Threshold 8 s sits between (margins ~1.4-2 s — thin;
# calibrated on one NG video). NOTE the other stretching style (normal-paced
# motions repeated/paused) stays inside the normal range and is NOT detectable
# by this metric.
TOO_LONG_SECONDS = 8.0
SLOW_GAP_TOL = 3.0        # gaps shorter than this do not break a slow episode


class InspectionWatchdog:
    """Feed each SAMPLED (~5 fps) feature vector; sustained_now() returns True
    while the sustained-inspection condition currently holds."""

    def __init__(self):
        data = np.load(MODEL_PATH)
        self.centroids = data["centroids"]          # (K, 55)
        self.sel = set(int(c) for c in data["sel"])
        # Slow-manipulation indicator: this cluster counts as inspection only
        # while local motion is low (careful cable turning vs brisk bench work).
        self.c_slow = int(data["c_slow"]) if "c_slow" in data else None
        self.motion_max = float(data["motion_max"]) if "motion_max" in data else 0.0
        self.reset()

    def reset(self):
        self.hits = deque()
        self.buf = deque(maxlen=11)   # (t, cluster, feat); CENTERED motion window
        # gate-6 slow-episode state (centered smoothing, like the offline validation)
        self.slow_flags = deque(maxlen=6)   # (t_mid, slow_hit)
        self.slow_ep_start = None
        self.slow_last = None
        self.slow_duration = 0.0            # running seconds of the current episode

    def sustained_now(self, feat, t):
        """Evaluate with a ~1 s decision delay: the sample judged is the MIDDLE
        of an 11-sample buffer, so the slow-motion test uses a CENTERED window —
        exactly the validated offline computation. (A trailing window mixed the
        preceding fast motion into slow-inspection onsets and broke detection.)
        The uniform ~1.1 s lag is negligible against the 13-16 s deadlines."""
        c = int(np.argmin(np.linalg.norm(self.centroids - feat[None, :], axis=1)))
        self.buf.append((t, c, feat))
        if len(self.buf) < self.buf.maxlen:
            return False
        tm, cm, _ = self.buf[len(self.buf) // 2]
        slow_hit = False
        if (self.c_slow is not None and cm == self.c_slow
                and motion_energy(np.array([f for _, _, f in self.buf])) < self.motion_max):
            slow_hit = True
        hit = (cm in self.sel) or slow_hit
        if hit:
            self.hits.append(tm)
        while self.hits and tm - self.hits[0] > SUSTAIN_WINDOW:
            self.hits.popleft()

        # --- gate-6 slow-episode tracking (centered smoothing) ----------------
        self.slow_flags.append((tm, slow_hit))
        if len(self.slow_flags) == self.slow_flags.maxlen:
            t_sig = self.slow_flags[len(self.slow_flags) // 2][0]
            sig = sum(1 for _, s in self.slow_flags if s) >= 2
            if sig:
                if self.slow_ep_start is None:
                    self.slow_ep_start = t_sig
                self.slow_last = t_sig
            elif (self.slow_ep_start is not None
                  and t_sig - self.slow_last > SLOW_GAP_TOL):
                dur = self.slow_last - self.slow_ep_start
                if dur >= 2.0:
                    print(f"  [info] slow-manipulation episode: {dur:.1f}s "
                          f"({self.slow_ep_start:.1f}-{self.slow_last:.1f}s)", flush=True)
                self.slow_ep_start = None
            # Duration = span of actual slow content (up to the LAST slow hit),
            # NOT up to the current time — otherwise the 3 s gap tolerance gets
            # counted into the episode and a 5.6 s episode reads as 8.6 s.
            self.slow_duration = (self.slow_last - self.slow_ep_start
                                  if self.slow_ep_start is not None else 0.0)

        return len(self.hits) >= SUSTAIN_MIN


def available():
    return os.path.exists(MODEL_PATH)
