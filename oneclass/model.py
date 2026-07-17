import torch
import torch.nn as nn

from pose_features import POSE_DIM, KP_DIM, NUM_KEYPOINTS

TOPK_FRACTION = 0.2   # anomaly score = mean of the top-20% per-timestep errors

# Dims used for the reconstruction loss & anomaly score: person-centric xy + bbox.
# The per-joint confidence channel is EXCLUDED — YOLO confidences flicker frame to
# frame (e.g. occluded lower body of a seated worker) and were measured to cause
# ~2/3 of the reconstruction error, drowning the actual motion signal. Confidence
# stays in the INPUT (the encoder can still use it to know which joints are valid);
# it is just not part of what must be rebuilt.
_kp = torch.arange(KP_DIM).reshape(NUM_KEYPOINTS, 3)
ERROR_DIMS = torch.cat([_kp[:, :2].reshape(-1), torch.arange(KP_DIM, POSE_DIM)])


def masked_mse(recon, x):
    """Training loss: MSE over ERROR_DIMS only."""
    idx = ERROR_DIMS.to(x.device)
    return ((recon - x)[..., idx] ** 2).mean()


class PoseLSTMAutoencoder(nn.Module):
    """Seq2seq LSTM autoencoder over pose windows, trained ONLY on normal work.

    The encoder compresses a 13 s pose window into a small latent vector; the
    decoder must rebuild the whole window from that vector alone. Because the
    bottleneck is tight and training data contains only correct work, the model
    can only rebuild motions it has seen: normal cycles reconstruct with low
    error, anything unlike normal work reconstructs poorly. The reconstruction
    error is therefore the anomaly score — no NG examples needed.

    Capacity note: keep hidden/latent SMALL. An over-sized autoencoder learns to
    copy any input (including anomalies), which destroys the anomaly signal.
    """

    def __init__(self, pose_dim=POSE_DIM, hidden_dim=128, latent_dim=32, num_layers=1):
        super().__init__()
        self.encoder = nn.LSTM(pose_dim, hidden_dim, num_layers, batch_first=True)
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.LSTM(latent_dim, hidden_dim, num_layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, pose_dim)

    def forward(self, x):
        """x: (B, S, pose_dim) -> reconstruction of the same shape."""
        B, S, _ = x.shape
        _, (hn, _) = self.encoder(x)
        z = self.to_latent(hn[-1])                     # (B, latent) bottleneck
        dec_in = z.unsqueeze(1).repeat(1, S, 1)        # latent repeated at each step
        out, _ = self.decoder(dec_in)
        return self.head(out)


def per_step_errors(model, x):
    """Per-timestep reconstruction MSE over ERROR_DIMS. x: (B, S, D) -> (B, S)."""
    with torch.no_grad():
        recon = model(x)
    idx = ERROR_DIMS.to(x.device)
    return ((recon - x)[..., idx] ** 2).mean(dim=2)


def score_from_errors(err):
    """Window anomaly score from per-timestep errors: mean of the top-20% steps.

    A plain mean dilutes brief anomalies (a ~2 s deviation is only ~15% of a 13 s
    window); a plain max is noisy. Top-k mean stays sensitive to short bursts
    while remaining stable. err: (B, S) -> (B,)
    """
    k = max(1, int(err.size(1) * TOPK_FRACTION))
    return torch.topk(err, k, dim=1).values.mean(dim=1)


def window_scores(model, x):
    """Convenience wrapper -> (scores (B,), per-step errors (B, S))."""
    err = per_step_errors(model, x)
    return score_from_errors(err), err
