"""Validate the finger-based pointing detector on the 3 NORMAL videos.

For gate-4 integration the detector must catch the pointing nearly every cycle
on normal footage (max event gap safely below the watchdog deadlines), otherwise
it will raise false NG alarms. Prints event times and gaps per video.
"""
import glob
import os
from collections import deque

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

ZX0, ZY0, ZS = 455, 440, 560
TIP_X_MIN = 740
IDX_MIN, OTH_MAX = 1.6, 1.3
STEP = 2
AGG_WINDOW, AGG_MIN, MERGE = 1.5, 2, 3.0
FINGERS = {"idx": (8, 6, 5), "mid": (12, 10, 9), "rng": (16, 14, 13), "pnk": (20, 18, 17)}

lmk = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=os.path.join(HERE, "hand_landmarker.task")),
    num_hands=2, min_hand_detection_confidence=0.2))


def fingers_of(hand):
    p = lambda i: np.array([hand[i].x, hand[i].y])
    return {n: float(np.linalg.norm(p(t) - p(m)) / max(np.linalg.norm(p(q) - p(m)), 1e-6))
            for n, (t, q, m) in FINGERS.items()}


videos = sorted(glob.glob(os.path.join(ROOT, "dataset", "normal", "*.mp4")))
videos.append(glob.glob(os.path.join(ROOT, "dataset", "abnormal", "no_pointing", "*.mp4"))[0])

for video in videos:
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    hits = deque()
    events = []
    last_event = -99.0
    fi = 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        fi += 1
        if fi % STEP:
            continue
        t = fi / fps
        crop = frame[ZY0:ZY0 + ZS, ZX0:ZX0 + ZS]
        big = cv2.resize(crop, (ZS * 2, ZS * 2))
        res = lmk.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                  data=cv2.cvtColor(big, cv2.COLOR_BGR2RGB)))
        for hand in (res.hand_landmarks or []):
            f = fingers_of(hand)
            if f["idx"] > IDX_MIN and max(f["mid"], f["rng"], f["pnk"]) < OTH_MAX:
                tipx = ZX0 + hand[8].x * ZS
                if tipx > TIP_X_MIN:
                    hits.append(t)
        while hits and t - hits[0] > AGG_WINDOW:
            hits.popleft()
        if len(hits) >= AGG_MIN and t - last_event > MERGE:
            events.append(round(t, 1))
            last_event = t
    dur = fi / fps
    cap.release()
    gaps = ([events[0]] + list(np.diff(events)) + [dur - events[-1]]) if events else [dur]
    name = os.path.basename(video)
    print(f"{name}: dur={dur:.0f}s events={len(events)} max_gap={max(gaps):.1f}s")
    print(f"  events @ {events}")
    print(f"  gaps>10s: {[round(float(x),1) for x in gaps if x > 10]}", flush=True)
