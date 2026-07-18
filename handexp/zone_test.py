"""Fixed work-zone hand tracking test.

Instead of cropping around the (unreliable) wrist keypoint, crop a FIXED region
of the work area (fixed camera) and run MediaPipe continuously over a segment
covering the confirmed pointing (32-34 s) plus surrounding work. Measures the
per-frame detection rate and the index-extension signal timeline.
"""
import os

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

VIDEO = os.path.join(ROOT, "dataset", "abnormal", "no_pointing", "2.検査ボード指差確認なし 1・2.mp4")
# Fixed work-zone crop (1920x1080): centered on where the hands operate
ZX0, ZY0, ZS = 455, 440, 560          # x0, y0, size
SEGMENTS = [(22.0, 26.0, "tray"), (31.0, 36.0, "pointing"), (44.0, 48.0, "next_cycle")]
STEP = 2                               # analyze every 2nd frame (15 fps)

lmk = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=os.path.join(HERE, "hand_landmarker.task")),
    num_hands=2, min_hand_detection_confidence=0.2))

CONN = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


def index_extension(hand):
    p = lambda i: np.array([hand[i].x, hand[i].y])
    return float(np.linalg.norm(p(8) - p(5)) / max(np.linalg.norm(p(6) - p(5)), 1e-6))


cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS) or 30

for t0, t1, label in SEGMENTS:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t0 * fps))
    n_frames = n_det = 0
    best = (0, None, None)   # (n_hands, t, annotated)
    exts = []
    fi = int(t0 * fps)
    while fi < int(t1 * fps):
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
        n = len(res.hand_landmarks or [])
        n_frames += 1
        if n:
            n_det += 1
            e = [round(index_extension(h), 2) for h in res.hand_landmarks]
            exts.append((round(t, 1), e))
            if n >= best[0]:
                ann = big.copy()
                h_, w_ = ann.shape[:2]
                for hand in res.hand_landmarks:
                    pts = [(int(p.x * w_), int(p.y * h_)) for p in hand]
                    for a, b in CONN:
                        cv2.line(ann, pts[a], pts[b], (0, 255, 0), 2)
                    for j, pt in enumerate(pts):
                        cv2.circle(ann, pt, 4, (0, 0, 255) if j == 8 else (255, 160, 0), -1)
                best = (n, t, ann)
    rate = n_det / max(1, n_frames) * 100
    print(f"[{label}] {t0}-{t1}s: hand detected in {n_det}/{n_frames} frames ({rate:.0f}%)")
    for t, e in exts:
        print(f"   t={t}s ext={e}")
    if best[2] is not None:
        cv2.imwrite(os.path.join(OUT, f"zone_{label}_{best[1]:.1f}s.jpg"),
                    cv2.resize(best[2], (640, 640)), [cv2.IMWRITE_JPEG_QUALITY, 88])
cap.release()
print(f"\nbest annotated crops -> {OUT}")
