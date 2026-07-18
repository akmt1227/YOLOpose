"""Refined pointing discriminator: index finger extended AND others curled.

Re-scans the same segments printing per-finger extension so we can verify that
true pointing separates from flat/open hands (tray) and fists.
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
SEGMENTS = [(22.0, 26.0, "tray/flat"), (31.0, 36.0, "POINTING"), (44.0, 48.0, "next POINTING")]
STEP = 2

FINGERS = {"idx": (8, 6, 5), "mid": (12, 10, 9), "rng": (16, 14, 13), "pnk": (20, 18, 17)}

lmk = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=os.path.join(HERE, "hand_landmarker.task")),
    num_hands=2, min_hand_detection_confidence=0.2))


def fingers_of(hand):
    p = lambda i: np.array([hand[i].x, hand[i].y])
    out = {}
    for name, (tip, pip_, mcp) in FINGERS.items():
        out[name] = float(np.linalg.norm(p(tip) - p(mcp)) / max(np.linalg.norm(p(pip_) - p(mcp)), 1e-6))
    return out


cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
for t0, t1, label in SEGMENTS:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t0 * fps))
    print(f"\n--- {label} ({t0}-{t1}s) ---   POINT? = idx>1.6 and others<1.3")
    fi = int(t0 * fps)
    while fi < int(t1 * fps):
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
            others = max(f["mid"], f["rng"], f["pnk"])
            is_point = f["idx"] > 1.6 and others < 1.3
            flag = "  <== POINTING" if is_point else ""
            print(f"  t={fi/fps:5.1f}s idx={f['idx']:.2f} mid={f['mid']:.2f} "
                  f"rng={f['rng']:.2f} pnk={f['pnk']:.2f}{flag}")
cap.release()
