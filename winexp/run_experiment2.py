"""Stride experiment: training-window stride 3 s (=15 samples, current) vs 1 s
(=5 samples), crossed with window lengths 6 s / 10 s / 13 s -> 6 configs.

Same data and metrics as run_experiment.py. Inference behavior is stride-
independent (detect.py decides every sample); this measures whether denser
training windows tighten the normal distribution / improve margins.
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
DROP_EVENTS = [3.0, 20.0]


def windows(S, L, stride):
    if len(S) < L:
        return np.empty((0, L, S.shape[1]), dtype=np.float32)
    return np.stack([S[i:i + L] for i in range(0, len(S) - L + 1, stride)]).astype(np.float32)


def score_all(model, W_):
    if len(W_) == 0:
        return np.array([])
    out = []
    with torch.no_grad():
        for i in range(0, len(W_), 512):
            s, _ = window_scores(model, torch.tensor(W_[i:i + 512]).to(DEVICE))
            out.append(s.cpu().numpy())
    return np.concatenate(out)


rows = []
for L, wl in [(30, "6s"), (50, "10s"), (65, "13s")]:
    for tr_stride, sl in [(15, "3s"), (5, "1s")]:
        torch.manual_seed(42)
        np.random.seed(42)
        X_tr = np.concatenate([windows(normal_series[0], L, tr_stride),
                               windows(normal_series[1], L, tr_stride)])
        X_cal = windows(normal_series[2], L, tr_stride)

        model = PoseLSTMAutoencoder().to(DEVICE)
        opt = optim.AdamW(model.parameters(), lr=1e-3)
        loader = DataLoader(TensorDataset(torch.tensor(X_tr)), batch_size=32, shuffle=True)
        # keep total gradient steps comparable: fewer epochs for the denser stride
        epochs = 300 if tr_stride == 15 else 100
        model.train()
        for ep in range(epochs):
            for (b,) in loader:
                b = b.to(DEVICE)
                opt.zero_grad()
                masked_mse(model(b), b).backward()
                opt.step()
        model.eval()

        cal = score_all(model, X_cal)
        thr = float(np.percentile(cal, 99))
        tight = thr / float(np.median(cal))

        Wd = windows(drop, L, 1)
        sd = score_all(model, Wd)
        t_end = (np.arange(len(sd)) + L) / FPS
        in_zone = np.zeros(len(sd), dtype=bool)
        ev = []
        for e in DROP_EVENTS:
            zone = (t_end >= e) & (t_end <= e + L / FPS)
            in_zone |= zone
            peak = sd[zone].max() if zone.any() else 0.0
            crossed = np.where(zone & (sd > thr))[0]
            lat = (t_end[crossed[0]] - e) if len(crossed) else None
            ev.append((peak / thr, lat))
        fp = float((sd[~in_zone] > thr).mean()) if (~in_zone).any() else 0.0

        Wt = windows(toolong, L, 1)
        st_ = score_all(model, Wt)
        above = st_ > thr
        ncross = int(np.sum(above[1:] & ~above[:-1]) + (1 if above[0] else 0))

        rows.append((wl, sl, len(X_tr), thr, tight, ev, fp, ncross))
        print(f"[win {wl} / stride {sl}] train={len(X_tr)} thr={thr:.4f} x{tight:.2f} | "
              f"drop m={ev[0][0]:.2f}/{ev[1][0]:.2f} lat={ev[0][1]}/{ev[1][1]} | "
              f"FP={fp*100:.1f}% | too_long x{ncross}", flush=True)

print("\n===== SUMMARY (win x stride) =====")
print("win  stride  train_n   thr    tight  drop@3s  drop@20s  FP%   too_long")
for wl, sl, n, thr, tight, ev, fp, nc in rows:
    print(f"{wl:>4} {sl:>6} {n:8d}  {thr:.4f}  {tight:4.2f}  "
          f"{ev[0][0]:6.2f}  {ev[1][0]:7.2f}  {fp*100:4.1f}  {nc:2d}x")
