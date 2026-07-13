import torch
import torch.nn as nn
import timm
from sentence_transformers import SentenceTransformer

class PoseActionLSTM(nn.Module):
    def __init__(self, pose_dim=17*2, visual_dim=256, hidden_dim=256, num_layers=2, embedding_dim=384):
        """
        LSTM model for anomaly detection.
        Combines YOLO keypoints with optional visual features from timm.
        
        Args:
            pose_dim: Dimension of YOLO keypoints (17 keypoints * 2 coordinates)
            visual_dim: Dimension of visual features extracted by timm (e.g. from human crop)
            hidden_dim: LSTM hidden states
            embedding_dim: Output embedding dimension (aligned to sentence-transformers 'all-MiniLM-L6-v2' -> 384)
        """
        super().__init__()
        # 1. Feature fusion (timm usage)
        self.visual_backbone = timm.create_model('mobilenetv3_small_050', pretrained=True, num_classes=visual_dim)
        
        lstm_input_dim = pose_dim + visual_dim
        
        # 2. Sequence Modeling with LSTM (PyTorch)
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # 3. Projection to text embedding space
        self.fc = nn.Linear(hidden_dim, embedding_dim)
        
    def extract_visual_features(self, crops):
        """
        Extract visual features using a timm backbone.
        crops: (batch_size, seq_len, C, H, W)
        """
        B, S, C, H, W = crops.shape
        crops = crops.view(B * S, C, H, W)
        features = self.visual_backbone(crops)
        return features.view(B, S, -1)
        
    def forward(self, keypoints, crops=None):
        """
        keypoints: (batch_size, seq_len, pose_dim)
        crops: (batch_size, seq_len, C, H, W)
        """
        B, S, _ = keypoints.shape
        
        if crops is not None:
            visual_feats = self.extract_visual_features(crops)
            x = torch.cat([keypoints, visual_feats], dim=-1)
        else:
            # If no visual crop is provided (for speed), pad with zeros
            visual_dim = self.visual_backbone.num_classes
            visual_feats = torch.zeros((B, S, visual_dim), device=keypoints.device)
            x = torch.cat([keypoints, visual_feats], dim=-1)
            
        out, (hn, cn) = self.lstm(x)
        # Take the hidden state of the last layer at the last time step
        last_hidden = hn[-1]
        
        # Project to sentence-transformer embedding size
        video_emb = self.fc(last_hidden)
        # L2 Normalize for metric learning
        video_emb = nn.functional.normalize(video_emb, p=2, dim=1)
        
        return video_emb
