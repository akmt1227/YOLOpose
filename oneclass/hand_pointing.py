"""Finger-based pointing-check detector for gate 4 (MediaPipe HandLandmarker).

Detects the actual pointing gesture — index finger extended, other fingers
curled, fingertip inside the pointing-target area — from a FIXED crop of the
work zone (fixed camera). This replaces the earlier pose-based proxy/strict
combination: it looks at the gesture itself, works despite white gloves, and
does not depend on the (unreliable-during-pointing) wrist keypoint.

Validated on the no-pointing NG video: events at 32.9 / 44.6 / 58.4 s matching
the operator ground truth and cycle cadence, zero events during the omission
span, button presses correctly rejected (their fingertip sits outside the
target area).

Camera/process-specific parameters (re-calibrate for a different station):
zone, TIP_X_MIN, finger thresholds.
"""
import os
import urllib.request
from collections import deque

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- fixed work zone (1920x1080 footage) and pointing-target area ------------
HAND_ZONE = (455, 440, 560)     # x0, y0, size of the analysis crop
TIP_X_MIN = 740                 # fingertip must be right of this (board/sheet side);
                                # true pointings measured at x 759-807, button presses <= 700

# --- finger-shape rule: index extended, other fingers curled -----------------
IDX_MIN, OTH_MAX = 1.6, 1.3
FINGERS = {"idx": (8, 6, 5), "mid": (12, 10, 9), "rng": (16, 14, 13), "pnk": (20, 18, 17)}

# --- temporal aggregation ----------------------------------------------------
# The real pointing check is a SUSTAINED 1-2 s gesture; with VIDEO-mode
# tracking it yields 14-26 hit-frames, while the false positives (index-out
# hand transiting the target area, e.g. toward the tray) top out at ~5.
# Requiring >=8 hit-frames within 2.5 s separates them with wide margin
# (measured on the no-pointing video with tracking enabled).
ANALYZE_HZ = 30                 # analyze every frame (CPU work)
AGG_WINDOW = 2.5                # >=AGG_MIN pointing-pose hit-frames within this window
AGG_MIN = 8                     # -> one pointing event
EVENT_MERGE = 4.0               # ignore further events for this long (suppresses
                                # re-triggering from a long gesture's own tail)

MODEL_NAME = "hand_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")

# Hand-skeleton drawing (per-finger chains + palm)
CONN = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
INDEX_CHAIN = [(5, 6), (6, 7), (7, 8)]


def _model_path():
    """Find (or download) the MediaPipe hand model."""
    for cand in (os.path.join(SCRIPT_DIR, MODEL_NAME),
                 os.path.join(os.path.dirname(SCRIPT_DIR), "handexp", MODEL_NAME)):
        if os.path.exists(cand):
            return cand
    dst = os.path.join(SCRIPT_DIR, MODEL_NAME)
    print(f"Downloading {MODEL_NAME} ...")
    urllib.request.urlretrieve(MODEL_URL, dst)
    return dst


def _fingers_of(hand):
    p = lambda i: np.array([hand[i].x, hand[i].y])
    return {n: float(np.linalg.norm(p(t) - p(m)) / max(np.linalg.norm(p(q) - p(m)), 1e-6))
            for n, (t, q, m) in FINGERS.items()}


class HandPointingDetector:
    """Feed every video frame; returns True on the frames where a pointing
    EVENT is registered (rising edge, merged within EVENT_MERGE seconds)."""

    def __init__(self, fps):
        self.fps = fps or 30.0
        self.interval = max(1, round(self.fps / ANALYZE_HZ))
        # VIDEO running mode: once a hand is found it is TRACKED across frames
        # (much steadier landmarks / less flicker than independent per-frame
        # detection, which matters for the small gloved hands in this footage).
        self.lmk = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=_model_path()),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2, min_hand_detection_confidence=0.2,
            min_tracking_confidence=0.3))
        self.reset()

    def reset(self):
        self.hits = deque()
        self.last_event_t = -1e9
        self.last_hands = []      # [(full-frame pts, is_pointing_pose)] of last analysis

    def process(self, frame, frame_idx):
        """Analyze (at ~ANALYZE_HZ) and return True if a pointing event fired now."""
        if frame_idx % self.interval:
            return False
        t = frame_idx / self.fps
        x0, y0, s = HAND_ZONE
        crop = frame[y0:y0 + s, x0:x0 + s]
        if crop.size == 0:
            return False
        big = cv2.resize(crop, (s * 2, s * 2))
        ts_ms = int(frame_idx / self.fps * 1000)   # VIDEO mode needs increasing timestamps
        res = self.lmk.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                                                 data=cv2.cvtColor(big, cv2.COLOR_BGR2RGB)), ts_ms)
        self.last_hands = []
        hit_now = False
        for hand in (res.hand_landmarks or []):
            f = _fingers_of(hand)
            pts = [(int(x0 + p.x * s), int(y0 + p.y * s)) for p in hand]
            shape_ok = f["idx"] > IDX_MIN and max(f["mid"], f["rng"], f["pnk"]) < OTH_MAX
            pointing = shape_ok and pts[8][0] > TIP_X_MIN
            self.last_hands.append((pts, pointing))
            if pointing:
                self.hits.append(t)
                hit_now = True
        while self.hits and t - self.hits[0] > AGG_WINDOW:
            self.hits.popleft()
        # Fire only on frames that ADDED a hit, so one long gesture cannot
        # re-trigger after EVENT_MERGE from its own stale window contents.
        if hit_now and len(self.hits) >= AGG_MIN and t - self.last_event_t > EVENT_MERGE:
            self.last_event_t = t
            return True
        return False

    def draw(self, frame):
        """Overlay the last hand skeleton(s) on an output frame: green bones,
        index chain in RED while the pointing pose is matched."""
        for pts, pointing in self.last_hands:
            for a, b in CONN:
                cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
            if pointing:
                for a, b in INDEX_CHAIN:
                    cv2.line(frame, pts[a], pts[b], (0, 0, 255), 4)
            for j, pt in enumerate(pts):
                cv2.circle(frame, pt, 3, (0, 0, 255) if j == 8 else (255, 160, 0), -1)
        return frame
