"""Train and SAVE the 6 s and 10 s AE models (same recipe as run_experiment.py)
into winexp/w6s/ and winexp/w10s/, in the format detect.py expects
(ae_model.pth + threshold.json), so annotated result videos can be produced.
"""
import json
import os
import sys

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "oneclass"))
from model import PoseLSTMAutoencoder, masked_mse, window_scores  # noqa: E402
from pose_features import STRIDE, motion_energy  # noqa: E402

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = False

HERE = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X = np.load(os.path.join(ROOT, "oneclass", "X_normal.npy"))
g = np.load(os.path.join(ROOT, "oneclass", "groups_normal.npy"))
series = []
for vid in np.unique(g):
    idx = np.where(g == vid)[0]
    s = [X[idx[0]]]
    for i in idx[1:]:
        s.append(X[i][-STRIDE:])
    series.append(np.vstack(s))


def windows(S, L, stride):
    return np.stack([S[i:i + L] for i in range(0, len(S) - L + 1, stride)]).astype(np.float32)


for L, name in [(30, "w6s"), (50, "w10s")]:
    torch.manual_seed(42)
    np.random.seed(42)
    X_tr = np.concatenate([windows(series[0], L, STRIDE), windows(series[1], L, STRIDE)])
    X_cal = windows(series[2], L, STRIDE)

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

    with torch.no_grad():
        cal, _ = window_scores(model, torch.tensor(X_cal).to(DEVICE))
    recon_thr = float(np.percentile(cal.cpu().numpy(), 99))
    energies = np.array([motion_energy(w) for w in X_tr])
    idle_thr = 0.5 * float(np.percentile(energies, 1))

    out = os.path.join(HERE, name)
    os.makedirs(out, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out, "ae_model.pth"))
    with open(os.path.join(out, "threshold.json"), "w") as f:
        json.dump({"recon_threshold": recon_thr, "idle_threshold": idle_thr,
                   "seq_len": L, "percentile": 99.0}, f, indent=2)
    print(f"{name}: L={L}  recon_thr={recon_thr:.4f}  idle_thr={idle_thr:.6f}  "
          f"train={len(X_tr)} cal={len(X_cal)}  -> saved")
