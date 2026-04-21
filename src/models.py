"""
TDT4265 - Model Architectures
Model 1: DenseNet121 (3D) via MONAI  — standard baseline
Model 2: EfficientNet-based MIL      — more advanced, uses 2D slices
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import DenseNet121


#Model 1: 3D DenseNet121 (MONAI built-in)

class BreastDenseNet(nn.Module):
    def __init__(self, in_channels=2, num_classes=3, dropout=0.3): # Changed to 2
        super().__init__()
        self.backbone = DenseNet121(
            spatial_dims=3,
            in_channels=in_channels, # Now accepting [Pre, Post_1]
            out_channels=num_classes,
            dropout_prob=dropout,
        )

    def forward(self, x):
        return self.backbone(x)


#Model 2: 2D EfficientNet + Multiple Instance Learning (MIL)

class SliceAttentionMIL(nn.Module):
    """
    Multiple Instance Learning over axial slices using EfficientNet-B0 as feature extractor.

    Idea:
      1. Extract N evenly-spaced 2D slices from the 3D volume.
      2. Encode each slice with a shared EfficientNet backbone.
      3. Aggregate slice embeddings with a learned attention mechanism.
      4. Classify the aggregated embedding.

    This approach is memory-efficient and leverages strong 2D ImageNet pretraining.
    """

    def __init__(self, num_classes: int = 3, num_slices: int = 16, dropout: float = 0.3, **kwargs):
        super().__init__()
        self.num_slices = num_slices

        try:
            import torchvision.models as tv
            backbone = tv.efficientnet_b0(weights="IMAGENET1K_V1")
            backbone.features[0][0] = nn.Conv2d(
                1, 32, kernel_size=3, stride=2, padding=1, bias=False
            )
            self.feature_dim = backbone.classifier[1].in_features
            backbone.classifier = nn.Identity()
            self.backbone = backbone
        except Exception:
            self.backbone = SimpleCNNEncoder()
            self.feature_dim = 256

        self.attention = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, num_classes),
        )

    def forward(self, x):
        """
        x: (B, C, H, W, D)  — 3D volume
        """
        B, C, H, W, D = x.shape

        indices = torch.linspace(0, D - 1, self.num_slices, dtype=torch.long)
        slices = x[:, :, :, :, indices]          # (B, C, H, W, num_slices)
        slices = slices.permute(0, 4, 1, 2, 3)   # (B, num_slices, C, H, W)
        slices = slices.reshape(B * self.num_slices, C, H, W)

        # Encode each slice
        feats = self.backbone(slices)             # (B*num_slices, feat_dim)
        feats = feats.view(B, self.num_slices, -1)  # (B, num_slices, feat_dim)

        # Attention pooling
        attn = self.attention(feats)              # (B, num_slices, 1)
        attn = F.softmax(attn, dim=1)
        aggregated = (attn * feats).sum(dim=1)    # (B, feat_dim)

        return self.classifier(aggregated)


class SimpleCNNEncoder(nn.Module):
    """Lightweight fallback 2D encoder if torchvision isn't available."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(128 * 16, 256), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


#Factory

def get_model(name: str, **kwargs) -> nn.Module:
    """
    name: 'densenet' | 'mil'
    """
    if name == "densenet":
        return BreastDenseNet(**kwargs)
    elif name == "mil":
        return SliceAttentionMIL(**kwargs)
    else:
        raise ValueError(f"Unknown model: {name}. Choose 'densenet' or 'mil'.")
