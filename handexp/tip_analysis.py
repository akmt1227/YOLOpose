"""Where does the index fingertip point during 'pointing-pose' hits?

Logs (t, tip_x, tip_y) full-frame for every pointing-pose hand over the whole
no-pointing video, so the true pointing target region (32-34 s, 44.7-45.7 s)
can be separated from button presses / other index-out shapes.
"""
import os

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
VIDEO = os.path.join(ROOT, "dataset", "abnormal", "no_pointing", "2.検査ボード指差確認なし 1・2.mp4")
ZX0, ZY0, ZS = 455, 440, 560
STEP = 2
FINGERS = {"idx": (8, 6, 5), "mid": (12, 10, 9), "rng": (16, 14, 13), "pnk": (20, 18, 17)}

lmk = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=os.path.join(HERE, "hand_landmarker.task")),
    num_hands=2, min_hand_detection_confidence=0.2))


def fingers_of(hand):
    p = lambda i: np.array([hand[i].x, hand[i].y])
    return {n: float(np.linalg.norm(p(t) - p(m)) / max(np.linalg.norm(p(q) - p(m)), 1e-6))
            for n, (t, q, m) in FINGERS.items()}


cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
fi = 0
hits = []
while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    fi += 1
    if fi % STEP:
        continue
    crop = frame[ZY0:ZY0 + ZS, ZX0:ZX0 + ZS]
    big = cv2.resize(crop, (ZS * 2, ZS * 2))
    res = lmk.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                              data=cv2.cvtColor(big, cv2.COLOR_BGR2RGB)))
    for hand in (res.hand_landmarks or []):
        f = fingers_of(hand)
        if f["idx"] > 1.6 and max(f["mid"], f["rng"], f["pnk"]) < 1.3:
            tip = hand[8]
            tx, ty = ZX0 + tip.x * ZS, ZY0 + tip.y * ZS
            hits.append((fi / fps, tx, ty))
cap.release()

print("t(s)   tip_x  tip_y   (true pointing: 32-34s, 44.7-45.7s)")
for t, x, y in hits:
    mark = " <== TRUE" if (32 <= t <= 34.9) or (44.5 <= t <= 46) else ""
    print(f"{t:5.1f}  {x:6.0f} {y:6.0f}{mark}")
