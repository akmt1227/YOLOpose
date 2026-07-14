import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sentence_transformers import SentenceTransformer
from models.anomaly_lstm import PoseActionLSTM
import numpy as np
import os
import json

# --- Windows / cuDNN LSTM workaround -----------------------------------------
# On Windows, PyTorch's cuDNN LSTM *backward* pass crashes the process during
# interpreter/DLL teardown (exit code 0xC0000409, STATUS_STACK_BUFFER_OVERRUN)
# *after* training finishes and the model is already saved. That non-zero exit
# made app.py report a false "training failed". Disabling cuDNN routes the LSTM
# through native CUDA kernels (still on GPU; negligibly slower for this small
# model) so the process exits cleanly. Inference (main.py) is forward-only and
# unaffected, so it keeps full cuDNN speed.
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = False

TEMPERATURE = 0.07
ACTIONS = [
    "person walking normally, standing, or doing routine activities",
    "person falling down, slipping, fainting, fighting, or moving abnormally",
]


def video_level_split(groups, labels, val_ratio=0.2, seed=42):
    """Stratified split by source video so windows from one video never span
    train and val, AND both classes are present on each side.

    Each source video is single-class (it came from dataset/normal or
    dataset/abnormal). We hold out a fraction of the videos *per class* for
    validation, always keeping >=1 video of that class in train and putting
    >=1 in val. This avoids a degenerate all-one-class validation set, which
    would make Val Acc and the margin calibration meaningless.

    Returns (train_mask, val_mask), or None if the split cannot guarantee both
    classes on both sides (fewer than 2 classes, or a class with only one
    video) -> the caller falls back to a random split with a warning.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    if len(uniq) < 2:
        return None

    # Each video id -> its (majority) class label; videos are single-class.
    group_label = {g: int(np.bincount(labels[groups == g]).argmax()) for g in uniq}

    by_class = {}
    for g in uniq:
        by_class.setdefault(group_label[g], []).append(g)

    # Need >=2 classes, each with >=2 videos, so every class can put >=1 video
    # in val while keeping >=1 in train.
    if len(by_class) < 2 or any(len(v) < 2 for v in by_class.values()):
        return None

    val_groups = []
    for gids in by_class.values():
        gids = np.array(gids)
        rng.shuffle(gids)
        n_val = min(max(1, int(round(len(gids) * val_ratio))), len(gids) - 1)
        val_groups.extend(gids[:n_val].tolist())

    val_mask = np.isin(groups, val_groups)
    return ~val_mask, val_mask


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    if not (os.path.exists("X_data.npy") and os.path.exists("y_labels.npy")):
        print("Data files not found. Please run prepare_data.py first.")
        return

    # Load extracted data
    X = np.load("X_data.npy")
    y = np.load("y_labels.npy")
    groups = np.load("groups.npy") if os.path.exists("groups.npy") else None

    # ---- Train / validation split (video-level to avoid data leakage) ----
    split = video_level_split(groups, y) if groups is not None else None
    if split is None:
        print("WARNING: cannot make a stratified video-level split (need >=2 videos per "
              "class; groups.npy may be missing) -> falling back to a random split. Add "
              "more videos per class and re-run prepare_data.py for a leak-free, balanced split.")
        rng = np.random.default_rng(42)
        idx = rng.permutation(len(X))
        n_val = max(1, int(len(X) * 0.2))
        val_mask = np.zeros(len(X), dtype=bool)
        val_mask[idx[:n_val]] = True
        train_mask = ~val_mask
    else:
        train_mask, val_mask = split

    X_tr = torch.tensor(X[train_mask], dtype=torch.float32)
    y_tr = torch.tensor(y[train_mask], dtype=torch.long)
    X_va = torch.tensor(X[val_mask], dtype=torch.float32)
    y_va = torch.tensor(y[val_mask], dtype=torch.long)

    print(f"Train sequences: {len(X_tr)} | Val sequences: {len(X_va)}")
    print(f"  Train normal/abnormal: {int((y_tr == 0).sum())}/{int((y_tr == 1).sum())}")
    print(f"  Val   normal/abnormal: {int((y_va == 0).sum())}/{int((y_va == 1).sum())}")

    # small batch size since we have small data
    dataloader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=16, shuffle=True)

    # Initialize our LSTM model
    model = PoseActionLSTM().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # Initialize Text Encoder for zero-shot alignment (Hugging Face). Kept frozen.
    print("Loading Text Encoder...")
    text_encoder = SentenceTransformer('all-MiniLM-L6-v2').to(device)
    with torch.no_grad():
        text_embs = text_encoder.encode(ACTIONS, convert_to_tensor=True).to(device)
        text_embs = F.normalize(text_embs, p=2, dim=1)  # Shape: (2, 384)

    num_epochs = 20
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            # Video embeddings from the LSTM (L2-normalized inside the model)
            video_embs = model(batch_X)                          # (B, 384)

            # CLIP-style objective: cosine similarity to the two frozen text anchors,
            # optimized with cross-entropy against the true class. This directly trains
            # the exact "normal vs abnormal" decision that inference (main.py) makes,
            # and pulls all same-class videos toward the same text anchor.
            logits = (video_embs @ text_embs.t()) / TEMPERATURE  # (B, 2)
            loss = F.cross_entropy(logits, batch_y)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        msg = f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/max(1, len(dataloader)):.4f}"

        # ---- Validation accuracy on held-out videos ----
        if len(X_va) > 0:
            model.eval()
            with torch.no_grad():
                val_logits = model(X_va.to(device)) @ text_embs.t()
                val_pred = val_logits.argmax(dim=1).cpu()
            acc = (val_pred == y_va).float().mean().item()
            msg += f", Val Acc: {acc*100:.1f}%"

        print(msg)

    # ---- Calibrate the decision margin on the validation set ----
    # Inference decides "abnormal" when (sim_abnormal - sim_normal) > margin.
    # Pick the margin that maximizes balanced accuracy on held-out videos, so
    # main.py doesn't rely on a hardcoded magic number.
    margin = 0.0
    if len(X_va) > 0:
        model.eval()
        with torch.no_grad():
            val_sims = model(X_va.to(device)) @ text_embs.t()      # (Nval, 2)
            scores = (val_sims[:, 1] - val_sims[:, 0]).cpu().numpy()
        y_np = y_va.numpy()
        pos, neg = (y_np == 1), (y_np == 0)

        uniq = np.unique(scores)
        mids = (uniq[:-1] + uniq[1:]) / 2 if len(uniq) > 1 else uniq
        candidates = np.concatenate([[scores.min() - 1e-3], mids, [scores.max() + 1e-3]])

        best_bacc = -1.0
        for t in candidates:
            pred = scores > t
            tpr = (pred & pos).sum() / max(1, pos.sum())
            tnr = (~pred & neg).sum() / max(1, neg.sum())
            bacc = 0.5 * (tpr + tnr)
            if bacc > best_bacc:
                best_bacc, margin = bacc, float(t)
        print(f"Calibrated decision margin: {margin:.4f} (val balanced acc: {best_bacc*100:.1f}%)")
    else:
        print("No validation set; saving default decision margin 0.0")

    with open("threshold.json", "w") as f:
        json.dump({"margin": margin}, f)
    print("Saved decision margin to threshold.json")

    # Save the trained model
    torch.save(model.state_dict(), "lstm_model.pth")
    print("Training complete! Model saved to lstm_model.pth")


if __name__ == "__main__":
    train()
