import torch
import torch.nn as nn


class PoseActionLSTM(nn.Module):
    def __init__(self, pose_dim=17*3, visual_dim=256, hidden_dim=256, num_layers=2,
                 embedding_dim=384, use_visual=False):
        """
        LSTM model for anomaly detection.

        By default this runs pose-only: the 51-dim per-frame keypoint features
        (17 keypoints * (x, y, confidence)) are fed straight into the LSTM. This is
        lean and fast, with no unused parameters.

        Optionally (use_visual=True) it fuses appearance features from a timm backbone
        run on each person crop. Enable this only once prepare_data.py / main.py are
        wired to supply crops of shape (B, S, C, H, W) to forward().

        Args:
            pose_dim: Dimension of pose features (17 keypoints * (x, y, confidence) = 51)
            visual_dim: Dimension of the timm visual features (only used if use_visual)
            hidden_dim: LSTM hidden size
            num_layers: Number of stacked LSTM layers
            embedding_dim: Output embedding size (aligned to 'all-MiniLM-L6-v2' -> 384)
            use_visual: Fuse timm visual features from person crops (default: False)
        """
        super().__init__()
        self.use_visual = use_visual
        self.visual_dim = visual_dim

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

        # Projection to the text embedding space
        self.fc = nn.Linear(hidden_dim, embedding_dim)

    def extract_visual_features(self, crops):
        """
        Extract visual features using the timm backbone.
        crops: (batch_size, seq_len, C, H, W)
        """
        B, S, C, H, W = crops.shape
        crops = crops.view(B * S, C, H, W)
        features = self.visual_backbone(crops)
        return features.view(B, S, -1)

    def forward(self, keypoints, crops=None):
        """
        keypoints: (batch_size, seq_len, pose_dim)
        crops:     (batch_size, seq_len, C, H, W), required only if use_visual=True
        """
        if self.use_visual:
            if crops is None:
                raise ValueError("use_visual=True requires `crops` of shape (B, S, C, H, W)")
            visual_feats = self.extract_visual_features(crops)
            x = torch.cat([keypoints, visual_feats], dim=-1)
        else:
            x = keypoints

        out, (hn, cn) = self.lstm(x)
        # Hidden state of the last layer at the last time step
        last_hidden = hn[-1]

        # Project to sentence-transformer embedding size and L2-normalize for metric learning
        video_emb = self.fc(last_hidden)
        video_emb = nn.functional.normalize(video_emb, p=2, dim=1)
        return video_emb
