"""Hand-keypoint feasibility test (MediaPipe Tasks HandLandmarker) on this footage.

Question: can we detect the worker's hand/fingers at all from this overhead
camera? Two modes per sampled moment:
  A) full-frame detection (hands are small -> likely to fail)
  B) wrist-ROI crops: YOLO pose gives the wrist positions -> crop & zoom around
     each wrist -> detect on the crop (the standard pipeline for small hands)

Saves annotated images into handexp/out/ for visual review and prints a summary.
"""
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "oneclass"))
from pose_features import resolve_yolo_weights, pick_worker  # noqa: E402
from ultralytics import YOLO  # noqa: E402

import mediapipe as mp  # noqa: E402
from mediapipe.tasks import python as mp_python  # noqa: E402
from mediapipe.tasks.python import vision  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

SAMPLES = [
    (os.path.join(ROOT, "dataset", "abnormal", "no_pointing", "2.検査ボード指差確認なし 1・2.mp4"), 33.5, "pointing"),
    (os.path.join(ROOT, "dataset", "abnormal", "no_pointing", "2.検査ボード指差確認なし 1・2.mp4"), 24.0, "tray_reach"),
    (os.path.join(ROOT, "dataset", "abnormal", "no_pointing", "2.検査ボード指差確認なし 1・2.mp4"), 40.0, "ordinary"),
    (os.path.join(ROOT, "dataset", "normal", "良品動画（テスト用）.mp4"), 14.4, "normal_gesture"),
    (os.path.join(ROOT, "dataset", "normal", "良品動画（テスト用）.mp4"), 45.0, "normal_work"),
]

CROP = 320
WRIST_KPS = [9, 10]

# Bone connections for drawing (per finger chains + palm)
CONN = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]

yolo = YOLO(resolve_yolo_weights())
lmk = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=os.path.join(HERE, "hand_landmarker.task")),
    num_hands=2, min_hand_detection_confidence=0.2))


def detect(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return lmk.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))


def annotate(img, res):
    h, w = img.shape[:2]
    for hand in (res.hand_landmarks or []):
        pts = [(int(p.x * w), int(p.y * h)) for p in hand]
        for a, b in CONN:
            cv2.line(img, pts[a], pts[b], (0, 255, 0), 2)
        for j, pt in enumerate(pts):
            color = (0, 0, 255) if j == 8 else (255, 160, 0)   # index tip in red
            cv2.circle(img, pt, 4, color, -1)
    return img


def index_extension(hand):
    """Rough 'index finger extended': tip(8) distance from mcp(5) vs pip(6)."""
    p = lambda i: np.array([hand[i].x, hand[i].y])
    return float(np.linalg.norm(p(8) - p(5)) / max(np.linalg.norm(p(6) - p(5)), 1e-6))


print(f"{'sample':>15} | full | crops(k9,k10) | index-ext")
for video, t, label in SAMPLES:
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print(f"{label:>15} | frame read failed")
        continue

    res_full = detect(frame)
    n_full = len(res_full.hand_landmarks or [])

    r = yolo(frame, verbose=False)
    crop_counts, ext_vals = [], []
    if r and len(r[0].boxes) > 0 and r[0].keypoints is not None:
        boxes = r[0].boxes.xyxy.cpu().numpy()
        wi, _ = pick_worker(boxes, None)
        kp = r[0].keypoints.xy.cpu().numpy()[wi]
        for k in WRIST_KPS:
            x, y = int(kp[k, 0]), int(kp[k, 1])
            x0, y0 = max(0, x - CROP // 2), max(0, y - CROP // 2)
            crop = frame[y0:y0 + CROP, x0:x0 + CROP].copy()
            if crop.size == 0:
                crop_counts.append(0)
                continue
            crop_big = cv2.resize(crop, (CROP * 2, CROP * 2))
            res_c = detect(crop_big)
            n_c = len(res_c.hand_landmarks or [])
            crop_counts.append(n_c)
            for hand in (res_c.hand_landmarks or []):
                ext_vals.append(round(index_extension(hand), 2))
            annotate(crop_big, res_c)
            cv2.imwrite(os.path.join(OUT, f"{label}_k{k}.jpg"), crop_big,
                        [cv2.IMWRITE_JPEG_QUALITY, 88])

    fullann = annotate(cv2.resize(frame, (1280, 720)), res_full)
    cv2.imwrite(os.path.join(OUT, f"{label}_full.jpg"), fullann, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"{label:>15} | {n_full:>4} | {str(crop_counts):>13} | {ext_vals}")

print(f"\nannotated images -> {OUT}")
