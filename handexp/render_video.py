"""Render the finger-pointing detector demo video using the PRODUCTION
HandPointingDetector (oneclass/hand_pointing.py) — VIDEO-mode tracking,
8-hit aggregation — so the demo always reflects the deployed behavior.

Overlay: work zone (blue) / pointing-target area (yellow) / hand skeletons
(green, index chain red on the pointing pose) / green POINTING DETECTED banner
on each registered event. CPU-only.
"""
import os
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "oneclass"))
from hand_pointing import HandPointingDetector, HAND_ZONE, TIP_X_MIN  # noqa: E402

VIDEO = os.path.join(ROOT, "dataset", "abnormal", "no_pointing", "2.検査ボード指差確認なし 1・2.mp4")
OUT = os.path.join(ROOT, "result_hand_test.mp4")
EVENT_HOLD = 2.0

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
writer = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

det = HandPointingDetector(fps)
zx, zy, zs = HAND_ZONE
event_until = -1.0
n_events = 0
fi = 0

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    t = fi / fps
    if det.process(frame, fi):
        n_events += 1
        event_until = t + EVENT_HOLD
        print(f"  [POINTING] event #{n_events} at ~{t:.1f}s", flush=True)

    cv2.rectangle(frame, (zx, zy), (zx + zs, zy + zs), (255, 140, 0), 2)
    cv2.putText(frame, "hand zone", (zx + 4, zy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 140, 0), 2)
    cv2.rectangle(frame, (TIP_X_MIN, zy), (zx + zs, zy + zs), (0, 220, 220), 1)
    cv2.putText(frame, "pointing target", (TIP_X_MIN + 4, zy + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 220), 1)
    det.draw(frame)
    if t <= event_until:
        cv2.rectangle(frame, (0, 0), (W, 70), (0, 160, 0), -1)
        cv2.putText(frame, f"POINTING DETECTED (#{n_events})", (24, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    writer.write(frame)
    fi += 1
    if fi % 600 == 0:
        print(f"  {fi}/{total}", flush=True)

cap.release()
writer.release()
print(f"done: {OUT}  events={n_events}")
