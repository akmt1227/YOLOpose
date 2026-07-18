"""Tip-stability analysis at FULL frame rate: is the real pointing distinguishable
from the tray-transit false hits by fingertip motion?"""
import glob
import os

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
VIDEO = glob.glob(os.path.join(ROOT, "dataset", "abnormal", "no_pointing", "2.*.mp4"))[0]
ZX0, ZY0, ZS = 455, 440, 560
TIP_X_MIN = 740
IDX_MIN, OTH_MAX = 1.6, 1.3
FINGERS = {"idx": (8, 6, 5), "mid": (12, 10, 9), "rng": (16, 14, 13), "pnk": (20, 18, 17)}
SEGMENTS = [(22.0, 24.5, "tray transit (false)"), (32.0, 35.5, "REAL pointing"),
            (42.5, 46.5, "REAL pointing 2"), (57.5, 60.0, "REAL pointing 3")]

lmk = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=os.path.join(HERE, "hand_landmarker.task")),
    num_hands=2, min_hand_detection_confidence=0.2))


def fingers_of(hand):
    p = lambda i: np.array([hand[i].x, hand[i].y])
    return {n: float(np.linalg.norm(p(t) - p(m)) / max(np.linalg.norm(p(q) - p(m)), 1e-6))
            for n, (t, q, m) in FINGERS.items()}


cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
for t0, t1, label in SEGMENTS:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t0 * fps))
    print(f"\n--- {label} ({t0}-{t1}s), every frame ---")
    hits = []
    fi = int(t0 * fps)
    while fi < int(t1 * fps):
        ok, frame = cap.read()
        if not ok:
            break
        fi += 1
        t = fi / fps
        crop = frame[ZY0:ZY0 + ZS, ZX0:ZX0 + ZS]
        big = cv2.resize(crop, (ZS * 2, ZS * 2))
        res = lmk.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                  data=cv2.cvtColor(big, cv2.COLOR_BGR2RGB)))
        for hand in (res.hand_landmarks or []):
            f = fingers_of(hand)
            if f["idx"] > IDX_MIN and max(f["mid"], f["rng"], f["pnk"]) < OTH_MAX:
                tx, ty = ZX0 + hand[8].x * ZS, ZY0 + hand[8].y * ZS
                if tx > TIP_X_MIN:
                    hits.append((t, tx, ty))
    if not hits:
        print("  (no hits)")
        continue
    for i, (t, x, y) in enumerate(hits):
        if i:
            dt = t - hits[i - 1][0]
            dd = ((x - hits[i - 1][1])**2 + (y - hits[i - 1][2])**2) ** 0.5
            spd = dd / max(dt, 1e-6)
            print(f"  t={t:6.2f}  tip=({x:5.0f},{y:5.0f})  d_prev={dd:5.1f}px  spd={spd:6.0f}px/s")
        else:
            print(f"  t={t:6.2f}  tip=({x:5.0f},{y:5.0f})")
cap.release()
