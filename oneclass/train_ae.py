"""Train the one-class autoencoder on NORMAL windows only, then calibrate thresholds.

No NG data is needed. Held-out normal videos give the score distribution of
correct work; the anomaly threshold is a high percentile of it. The idle
threshold comes from the low tail of the normal motion-energy distribution.

Output (into --work_dir): ae_model.pth, threshold.json
"""
import os
import json
import argparse

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from model import PoseLSTMAutoencoder, window_scores, masked_mse
from pose_features import motion_energy

# Windows / cuDNN LSTM workaround: the cuDNN LSTM backward pass can crash the
# process at teardown (exit 0xC0000409) after training already finished, making
# the GUI report a false failure. Native CUDA kernels avoid it (same GPU speed
# class for this small model).
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def video_level_split(groups, val_ratio=0.2, seed=42):
    """Hold out whole videos for calibration so train/val windows never share a video."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    if len(uniq) < 2:
        return None
    rng.shuffle(uniq)
    n_val = min(max(1, int(round(len(uniq) * val_ratio))), len(uniq) - 1)
    val_groups = set(uniq[:n_val].tolist())
    val_mask = np.isin(groups, list(val_groups))
    return ~val_mask, val_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', type=str, default=SCRIPT_DIR)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--percentile', type=float, default=99.0,
                        help='Anomaly threshold = this percentile of held-out normal scores. '
                             'Lower it (e.g. 95) for higher sensitivity / more false alarms.')
    args = parser.parse_args()

    x_path = os.path.join(args.work_dir, 'X_normal.npy')
    g_path = os.path.join(args.work_dir, 'groups_normal.npy')
    if not os.path.exists(x_path):
        print(f"{x_path} not found. Run prepare_normal.py first.")
        return

    X = np.load(x_path)
    groups = np.load(g_path) if os.path.exists(g_path) else np.zeros(len(X), dtype=np.int64)

    split = video_level_split(groups)
    if split is None:
        print("WARNING: only 1 normal video -> falling back to a window-level split. "
              "Add more normal videos for a reliable threshold.")
        rng = np.random.default_rng(42)
        idx = rng.permutation(len(X))
        n_val = max(1, int(len(X) * 0.2))
        val_mask = np.zeros(len(X), dtype=bool)
        val_mask[idx[:n_val]] = True
        split = (~val_mask, val_mask)
    train_mask, val_mask = split

    X_tr = torch.tensor(X[train_mask], dtype=torch.float32)
    X_va = torch.tensor(X[val_mask], dtype=torch.float32)
    print(f"Train windows: {len(X_tr)} | Calibration (held-out normal) windows: {len(X_va)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    model = PoseLSTMAutoencoder().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(X_tr), batch_size=32, shuffle=True)

    print("Training autoencoder on normal work only...")
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            # Masked MSE: rebuild pose xy + bbox; the noisy conf channel is excluded.
            loss = masked_mse(model(batch), batch)
            loss.backward()
            optimizer.step()
            total += loss.item()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}], Recon Loss: {total/max(1, len(loader)):.6f}")

    # ---- Calibrate thresholds ----
    model.eval()
    val_scores, _ = window_scores(model, X_va.to(device))
    val_scores = val_scores.cpu().numpy()
    recon_threshold = float(np.percentile(val_scores, args.percentile))

    energies = np.array([motion_energy(w) for w in X[train_mask]])
    idle_threshold = 0.5 * float(np.percentile(energies, 1))

    print("Held-out normal score distribution:")
    print(f"  mean={val_scores.mean():.5f}  p50={np.percentile(val_scores, 50):.5f}  "
          f"p95={np.percentile(val_scores, 95):.5f}  max={val_scores.max():.5f}")
    print(f"Anomaly threshold (p{args.percentile:g}): {recon_threshold:.5f}")
    print(f"Idle threshold (motion energy):        {idle_threshold:.6f} "
          f"(normal p1={np.percentile(energies, 1):.6f}, median={np.median(energies):.6f})")

    torch.save(model.state_dict(), os.path.join(args.work_dir, 'ae_model.pth'))
    with open(os.path.join(args.work_dir, 'threshold.json'), 'w') as f:
        json.dump({
            'recon_threshold': recon_threshold,
            'idle_threshold': idle_threshold,
            'percentile': args.percentile,
            'val_scores': {'mean': float(val_scores.mean()),
                           'p95': float(np.percentile(val_scores, 95)),
                           'max': float(val_scores.max())},
        }, f, indent=2)
    print(f"Saved ae_model.pth and threshold.json to {args.work_dir}")


if __name__ == '__main__':
    main()
