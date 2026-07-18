"""Window-length experiment: train & evaluate the one-class AE at 6 s / 10 s / 13 s.

Data (all continuous 5 fps feature series, no YOLO needed here):
  - 3 normal videos: rebuilt from oneclass/X_normal.npy (train = videos 0+1,
    calibration = video 2, matching the production setup)
  - series_drop.npz     (drops at ~3 s and ~20 s, operator ground truth)
  - series_toolong.npz  (stretched inspection cycles)

Metrics per window length L:
  - threshold p99 of held-out normal scores, and how tight normal scores sit
  - drop events: detected? peak-score/threshold margin, alert latency after event
  - false-positive rate on drop video outside the event zones
  - too_long: number/time of threshold crossings (cycle-context sensitivity)
  - first possible decision time (= L / 5 fps, cold start)
"""
import os
import sys

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "oneclass"))
from model import PoseLSTMAutoencoder, masked_mse, window_scores  # noqa: E402
from pose_features import STRIDE  # noqa: E402

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = False

FPS = 5.0
HERE = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- rebuild normal series from the production windows ----
X = np.load(os.path.join(ROOT, "oneclass", "X_normal.npy"))
g = np.load(os.path.join(ROOT, "oneclass", "groups_normal.npy"))
normal_series = []
for vid in np.unique(g):
    idx = np.where(g == vid)[0]
    s = [X[idx[0]]]
    for i in idx[1:]:
        s.append(X[i][-STRIDE:])
    normal_series.append(np.vstack(s))

drop = np.load(os.path.join(HERE, "series_drop.npz"))["series"]
toolong = np.load(os.path.join(HERE, "series_toolong.npz"))["series"]
print(f"series: normals {[len(s) for s in normal_series]}  drop {len(drop)}  toolong {len(toolong)}")

DROP_EVENTS = [3.0, 20.0]


def windows(S, L, stride):
    if len(S) < L:
        return np.empty((0, L, S.shape[1]), dtype=np.float32)
    return np.stack([S[i:i + L] for i in range(0, len(S) - L + 1, stride)]).astype(np.float32)


def score_all(model, W_):
    if len(W_) == 0:
        return np.array([])
    with torch.no_grad():
        s, _ = window_scores(model, torch.tensor(W_).to(DEVICE))
    return s.cpu().numpy()


results = []
for L, label in [(30, "6s"), (50, "10s"), (65, "13s")]:
    torch.manual_seed(42)
    np.random.seed(42)

    X_tr = np.concatenate([windows(normal_series[0], L, STRIDE),
                           windows(normal_series[1], L, STRIDE)])
    X_cal = windows(normal_series[2], L, STRIDE)

    model = PoseLSTMAutoencoder().to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(torch.tensor(X_tr)), batch_size=32, shuffle=True)
    model.train()
    for ep in range(300):
        for (b,) in loader:
            b = b.to(DEVICE)
            opt.zero_grad()
            masked_mse(model(b), b).backward()
            opt.step()
    model.eval()

    cal = score_all(model, X_cal)
    thr = float(np.percentile(cal, 99))
    tightness = thr / float(np.median(cal))

    # drop timeline: sliding stride 1, window END time = (i+L)/FPS
    Wd = windows(drop, L, 1)
    sd = score_all(model, Wd)
    t_end = (np.arange(len(sd)) + L) / FPS

    ev_rows = []
    in_zone = np.zeros(len(sd), dtype=bool)
    for e in DROP_EVENTS:
        zone = (t_end >= e) & (t_end <= e + L / FPS)
        in_zone |= zone
        peak = sd[zone].max() if zone.any() else 0.0
        crossed = np.where(zone & (sd > thr))[0]
        latency = (t_end[crossed[0]] - e) if len(crossed) else None
        ev_rows.append((e, peak / thr, latency))
    fp = float((sd[~in_zone] > thr).mean()) if (~in_zone).any() else 0.0

    # too_long: distinct crossings
    Wt = windows(toolong, L, 1)
    st_ = score_all(model, Wt)
    tt = (np.arange(len(st_)) + L) / FPS
    above = st_ > thr
    crossings = []
    prev = False
    for j in range(len(above)):
        if above[j] and not prev:
            crossings.append(round(float(tt[j]), 1))
        prev = above[j]

    results.append((label, L, thr, tightness, ev_rows, fp, crossings))
    print(f"\n[{label}] thr={thr:.4f} (p99; x{tightness:.2f} of median)  first decision @ {L/FPS:.0f}s")
    for e, m, lat in ev_rows:
        print(f"  drop@{e:.0f}s: peak/thr = {m:.2f}  latency = "
              + (f"{lat:.1f}s" if lat is not None else "NOT DETECTED"))
    print(f"  drop FP rate outside events: {fp*100:.1f}%")
    print(f"  too_long crossings ({len(crossings)}): {crossings}")

print("\n===== SUMMARY =====")
print("win   thr     drop@3s(m,lat)   drop@20s(m,lat)   FP%   too_long")
for label, L, thr, tight, ev, fp, cr in results:
    e3 = f"{ev[0][1]:.2f}," + (f"{ev[0][2]:.1f}s" if ev[0][2] is not None else "MISS")
    e20 = f"{ev[1][1]:.2f}," + (f"{ev[1][2]:.1f}s" if ev[1][2] is not None else "MISS")
    print(f"{label:>4}  {thr:.4f}  {e3:>14}  {e20:>15}  {fp*100:4.1f}  {len(cr)} crossings")
