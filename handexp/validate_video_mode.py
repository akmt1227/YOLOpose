"""Validate the PRODUCTION HandPointingDetector (now VIDEO mode) on the
no-pointing video: detection rate, pointing-pose hit timeline, event times.

Pass/fail: events only at the true pointings (~33-35, ~44-46, ~58-59 s),
NONE during the omission span (0-30 s, esp. the ~23 s tray transit).
"""
import glob
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "oneclass"))
from hand_pointing import HandPointingDetector  # noqa: E402

video = glob.glob(os.path.join(ROOT, "dataset", "abnormal", "no_pointing", "2.*.mp4"))[0]
cap = cv2.VideoCapture(video)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
det = HandPointingDetector(fps)

events = []
n_frames = n_hand = 0
hit_times = []
fi = 0
prev_hits = 0
while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    if det.process(frame, fi):
        events.append(round(fi / fps, 1))
    if fi % det.interval == 0:
        n_frames += 1
        if det.last_hands:
            n_hand += 1
        if len(det.hits) > prev_hits:
            hit_times.append(round(fi / fps, 2))
    prev_hits = len(det.hits)
    fi += 1
cap.release()

print(f"hand detection rate: {n_hand}/{n_frames} analyzed frames ({n_hand/max(1,n_frames)*100:.0f}%)")
print(f"pointing-pose hits ({len(hit_times)}): {hit_times}")
print(f"EVENTS: {events}")
bad = [e for e in events if e < 30]
print("VERDICT:", "FAIL - events during omission span:" + str(bad) if bad else
      "PASS - no events during the omission span")
