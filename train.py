import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from models.anomaly_lstm import PoseActionLSTM
from utils.config import CLASSES
import numpy as np
import os
import json


def video_level_split(groups, y, val_ratio=0.2, seed=42):
    """Stratified split by source video.

    Holds out a fraction of EACH class's videos for validation, so both train and
    val contain every class that has >=2 videos, and windows from one video never
    span train and val. Returns (train_mask, val_mask), or None if no class has
    >=2 videos to split.
    """
    rng = np.random.default_rng(seed)
    val_groups = set()
    for cls in np.unique(y):
        cls_groups = np.unique(groups[y == cls])
        if len(cls_groups) < 2:
            continue  # only 1 video of this class -> keep it all for training
        rng.shuffle(cls_groups)
        n_val = min(max(1, int(round(len(cls_groups) * val_ratio))), len(cls_groups) - 1)
        val_groups.update(cls_groups[:n_val].tolist())
    if not val_groups:
        return None
    val_mask = np.isin(groups, list(val_groups))
    return ~val_mask, val_mask


def stratified_random_split(y, val_ratio=0.2, seed=42):
    """Fallback split (windows, not videos) that still keeps every class in val."""
    rng = np.random.default_rng(seed)
    val_mask = np.zeros(len(y), dtype=bool)
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * val_ratio))
        val_mask[idx[:n_val]] = True
    return ~val_mask, val_mask


def calibrate_threshold(probs, y_binary):
    """Pick the P(NG) threshold maximizing balanced accuracy on validation."""
    pos, neg = (y_binary == 1), (y_binary == 0)
    uniq = np.unique(probs)
    mids = (uniq[:-1] + uniq[1:]) / 2 if len(uniq) > 1 else uniq
    candidates = np.concatenate([[0.0], mids, [1.0]])
    best_bacc, best_t = -1.0, 0.5
    for t in candidates:
        pred = probs > t
        tpr = (pred & pos).sum() / max(1, pos.sum())
        tnr = (~pred & neg).sum() / max(1, neg.sum())
        bacc = 0.5 * (tpr + tnr)
        if bacc > best_bacc:
            best_bacc, best_t = bacc, float(t)
    return best_t, best_bacc


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    if not (os.path.exists("X_data.npy") and os.path.exists("y_labels.npy")):
        print("Data files not found. Please run prepare_data.py first.")
        return

    X = np.load("X_data.npy")
    y = np.load("y_labels.npy")
    groups = np.load("groups.npy") if os.path.exists("groups.npy") else None

    num_classes = len(CLASSES)
    for cls_idx, name in enumerate(CLASSES):
        if np.sum(y == cls_idx) == 0:
            print(f"WARNING: class '{name}' has no data — it cannot be learned. "
                  f"Add videos to its folder and re-run prepare_data.py.")

    # ---- Train / validation split (stratified, video-level to avoid leakage) ----
    split = video_level_split(groups, y) if groups is not None else None
    if split is None:
        print("WARNING: no class has >=2 videos (or groups.npy missing) -> falling back to "
              "a window-level split. Add more videos per class for a leak-free split.")
        split = stratified_random_split(y)
    train_mask, val_mask = split

    X_tr = torch.tensor(X[train_mask], dtype=torch.float32)
    y_tr = torch.tensor(y[train_mask], dtype=torch.long)
    X_va = torch.tensor(X[val_mask], dtype=torch.float32)
    y_va = torch.tensor(y[val_mask], dtype=torch.long)

    print(f"Train sequences: {len(X_tr)} | Val sequences: {len(X_va)}")
    for cls_idx, name in enumerate(CLASSES):
        print(f"  {name}: train {int((y_tr == cls_idx).sum())} / val {int((y_va == cls_idx).sum())}")

    dataloader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=16, shuffle=True)

    model = PoseActionLSTM().to(device)   # multi-class classifier, pose-only
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # Class weights counter imbalance between OK and the (smaller) NG classes.
    class_counts = np.bincount(y[train_mask], minlength=num_classes).astype(np.float32)
    weights = torch.tensor(len(y_tr) / (num_classes * np.maximum(class_counts, 1.0)),
                           dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    num_epochs = 30
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_X)                 # (B, num_classes)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        msg = f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/max(1, len(dataloader)):.4f}"
        if len(X_va) > 0:
            model.eval()
            with torch.no_grad():
                pred = model(X_va.to(device)).argmax(dim=1).cpu()
            acc = (pred == y_va).float().mean().item()
            msg += f", Val Acc: {acc*100:.1f}%"
        print(msg)

    # ---- Final validation report: per-class recall ----
    val_probs = None
    if len(X_va) > 0:
        model.eval()
        with torch.no_grad():
            val_probs = F.softmax(model(X_va.to(device)), dim=1).cpu().numpy()
        pred = val_probs.argmax(axis=1)
        y_np = y_va.numpy()
        print("Per-class recall on held-out videos:")
        for cls_idx, name in enumerate(CLASSES):
            n = int((y_np == cls_idx).sum())
            if n == 0:
                print(f"  {name}: (no val samples)")
            else:
                r = ((pred == cls_idx) & (y_np == cls_idx)).sum() / n
                print(f"  {name}: {r*100:.1f}% ({n} samples)")

    # ---- Calibrate the NG threshold: NG if P(NG) = 1 - P(normal) > threshold ----
    # Needs both OK and NG samples in val to be meaningful.
    threshold = 0.5
    if val_probs is not None:
        y_binary = (y_va.numpy() != 0).astype(int)
        if len(np.unique(y_binary)) == 2:
            p_ng = 1.0 - val_probs[:, 0]
            threshold, bacc = calibrate_threshold(p_ng, y_binary)
            print(f"Calibrated NG threshold: P(NG) > {threshold:.3f} (val balanced acc: {bacc*100:.1f}%)")
        else:
            print("Val set lacks OK or NG samples; keeping default NG threshold 0.5.")
    else:
        print("No validation set; saving default NG threshold 0.5.")

    with open("threshold.json", "w") as f:
        json.dump({"threshold": threshold}, f)
    print("Saved NG threshold to threshold.json")

    torch.save(model.state_dict(), "lstm_model.pth")
    print("Training complete! Model saved to lstm_model.pth")


if __name__ == "__main__":
    train()
