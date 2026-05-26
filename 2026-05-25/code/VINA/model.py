"""
VINA Model Architecture

Implements a unified AIGC detector that processes both images and video frames.
Backbone: lightweight ResNet-like CNN (can be swapped for ViT/DINOv2 in production)
Heads:
  - Classification head: binary real/fake
  - Projection head: for cross-modal supervised contrastive loss

Paper: "Video as Natural Augmentation: Towards Unified AI-Generated Image and Video Detection"
arXiv: 2605.21977
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = ConvBlock(ch, ch)
        self.c2 = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch)
        )

    def forward(self, x):
        return F.relu(x + self.c2(self.c1(x)), inplace=True)


class VINAEncoder(nn.Module):
    """
    Lightweight CNN encoder shared for both image and video frame inputs.
    In the paper, a pre-trained vision backbone (e.g., ViT-B) is used.
    Here we use a small ResNet-like CNN for the toy reproduction.
    """

    def __init__(self, in_channels=3, base_ch=32, embed_dim=128):
        super().__init__()
        self.stem = ConvBlock(in_channels, base_ch, stride=2)   # 64 → 32
        self.layer1 = nn.Sequential(
            ConvBlock(base_ch, base_ch * 2, stride=2),          # 32 → 16
            ResidualBlock(base_ch * 2)
        )
        self.layer2 = nn.Sequential(
            ConvBlock(base_ch * 2, base_ch * 4, stride=2),      # 16 → 8
            ResidualBlock(base_ch * 4)
        )
        self.layer3 = nn.Sequential(
            ConvBlock(base_ch * 4, base_ch * 8, stride=2),      # 8 → 4
            ResidualBlock(base_ch * 8)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(base_ch * 8, embed_dim)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return self.proj(x)  # (B, embed_dim)


class VINADetector(nn.Module):
    """
    VINA Unified AIGC Detector.

    Shared encoder processes both image and video frame inputs.
    Two heads:
      1. Classification head: binary real/fake detection
      2. Contrastive projection head: for cross-modal supervised contrastive loss

    The key insight: during training, pairs of (image, video_frame) from the same
    source are used. The contrastive loss aligns their representations in embedding
    space, bridging the cross-modal gap introduced by video codec processing.
    """

    def __init__(self, embed_dim=128, proj_dim=64, num_classes=2):
        super().__init__()
        self.encoder = VINAEncoder(embed_dim=embed_dim)

        # Classification head (used for inference)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(embed_dim // 2, num_classes)
        )

        # Projection head for contrastive learning (discarded at inference)
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, proj_dim)
        )

    def forward(self, x, return_proj=False):
        """
        Args:
            x: (B, C, H, W) image or video frame tensor
            return_proj: if True, also return normalized projection for contrastive loss
        Returns:
            logits: (B, num_classes)
            proj (optional): (B, proj_dim) L2-normalized projection
        """
        feat = self.encoder(x)
        logits = self.classifier(feat)
        if return_proj:
            proj = F.normalize(self.projector(feat), dim=-1)
            return logits, proj
        return logits

    def encode(self, x):
        """Extract normalized embeddings (for evaluation/retrieval)."""
        feat = self.encoder(x)
        return F.normalize(feat, dim=-1)


class CrossModalContrastiveLoss(nn.Module):
    """
    Cross-Modal Supervised Contrastive Loss for VINA.

    Given paired (image_proj, video_proj) with labels, brings same-source
    (real-real or fake-fake) image-video pairs together while pushing
    different-source pairs apart.

    Based on: SupCon loss extended to cross-modal pairs.

    L_SCL = -1/N * sum_i log [
        exp(z_img_i · z_vid_i / τ) /
        sum_j exp(z_img_i · z_vid_j / τ)
    ]
    where positive pairs are same-label (i==j), negatives are different-label.
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, img_proj, vid_proj, labels):
        """
        Args:
            img_proj: (B, D) L2-normalized image projections
            vid_proj: (B, D) L2-normalized video frame projections
            labels: (B,) binary labels (0=real, 1=fake)
        Returns:
            scalar loss
        """
        B = img_proj.size(0)
        # Cross-modal similarity matrix: img vs video
        sim = torch.matmul(img_proj, vid_proj.T) / self.temperature  # (B, B)

        # Positive mask: same label (same source)
        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.T).float()  # (B, B)

        # Numerator: exp(sim for positive pairs)
        exp_sim = torch.exp(sim)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # Mean over positive pairs
        loss = -(pos_mask * log_prob).sum(dim=1) / (pos_mask.sum(dim=1) + 1e-8)
        return loss.mean()
