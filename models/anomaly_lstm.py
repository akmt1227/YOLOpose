import torch
import torch.nn as nn

from utils.pose import POSE_DIM
from utils.config import CLASSES


class PoseActionLSTM(nn.Module):
    def __init__(self, pose_dim=POSE_DIM, visual_dim=256, hidden_dim=256, num_layers=2,
                 num_classes=len(CLASSES), dropout=0.3, use_visual=False):
        """
        LSTM classifier for work-motion NG detection (multi-class).

        Consumes the pose-feature sequence of one inspection cycle (~13 s) and outputs
        class logits: index 0 = normal (OK), the rest = NG types (utils/config.py:CLASSES).

        The sequence is summarized by **temporal attention pooling** over the LSTM
        outputs instead of just the last hidden state:
          - brief events (e.g. dropping the part) appear as a few salient time steps
            that attention picks up directly, without being diluted by the rest of
            the window;
          - omissions (e.g. missing pointing check, skipped inspection area) show up
            in the whole-window aggregate;
          - the attention weights also tell WHEN in the window the model looked,
            which main.py uses to explain NG alerts.

        Runs pose-only by default. Set use_visual=True to also fuse timm appearance
        features from person crops (requires crops of shape (B, S, C, H, W) in forward()).

        Args:
            pose_dim: Pose feature size (51 person-centric + 4 absolute bbox = 55)
            visual_dim: timm visual feature size (only used if use_visual)
            hidden_dim: LSTM hidden size
            num_layers: Number of stacked LSTM layers
            num_classes: Output classes (index 0 = normal)
            dropout: Dropout before the classifier head
            use_visual: Fuse timm visual features from person crops (default: False)
        """
        super().__init__()
        self.use_visual = use_visual
        self.visual_dim = visual_dim
        self.num_classes = num_classes

        if use_visual:
            # Lazy import so pose-only users don't need timm installed.
            import timm
            self.visual_backbone = timm.create_model(
                'mobilenetv3_small_050', pretrained=True, num_classes=visual_dim
            )
            lstm_input_dim = pose_dim + visual_dim
        else:
            self.visual_backbone = None
            lstm_input_dim = pose_dim

        # Sequence modeling with LSTM (PyTorch)
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0,
        )

        # Temporal attention pooling over the LSTM outputs
        self.attn = nn.Linear(hidden_dim, 1)

        # Classifier head
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def extract_visual_features(self, crops):
        """
        Extract visual features using the timm backbone.
        crops: (batch_size, seq_len, C, H, W)
        """
        B, S, C, H, W = crops.shape
        crops = crops.view(B * S, C, H, W)
        features = self.visual_backbone(crops)
        return features.view(B, S, -1)

    def forward(self, keypoints, crops=None, return_attn=False):
        """
        keypoints: (batch_size, seq_len, pose_dim)
        crops:     (batch_size, seq_len, C, H, W), required only if use_visual=True
        return_attn: also return the temporal attention weights (batch_size, seq_len)

        returns: (batch_size, num_classes) class logits
                 [, (batch_size, seq_len) attention weights if return_attn]
        """
        if self.use_visual:
            if crops is None:
                raise ValueError("use_visual=True requires `crops` of shape (B, S, C, H, W)")
            visual_feats = self.extract_visual_features(crops)
            x = torch.cat([keypoints, visual_feats], dim=-1)
        else:
            x = keypoints

        out, _ = self.lstm(x)                              # (B, S, H)

        # Attention pooling: weight each time step, then aggregate.
        attn_w = torch.softmax(self.attn(out), dim=1)      # (B, S, 1), softmax over time
        ctx = (attn_w * out).sum(dim=1)                    # (B, H)

        logits = self.fc(self.dropout(ctx))
        if return_attn:
            return logits, attn_w.squeeze(-1)
        return logits
