"""
Supervised Classification Pipeline (Pipeline A).

Paper (§3.1): Supervised multiclass classification for known violation categories.
Boosted by MLLM knowledge distillation (§3.3).

Architecture: Lightweight student model trained with:
  - Cross-entropy loss on ground-truth labels
  - KL-divergence loss on MLLM teacher soft labels (knowledge distillation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationStudent(nn.Module):
    """
    Lightweight student classifier (MLLM-distilled).

    Paper: "a multimodal large language model (MLLM) distilling knowledge
    into each [pipeline] to boost accuracy while keeping inference lightweight."

    Input: fused multimodal features
    Output: violation class logits
    """

    def __init__(self, audio_dim=512, visual_dim=768, text_dim=128,
                 hidden_dim=256, num_classes=2):
        super().__init__()
        # Lightweight projection layers (vs. teacher's 1024-d)
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.ReLU(),
        )
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.ReLU(),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
        )

        # Multimodal fusion via attention pooling
        self.fusion_attn = nn.Linear(hidden_dim, 1)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, text_feat, audio_feat, visual_feat):
        """
        Args:
            text_feat:   (B, text_dim)
            audio_feat:  (B, audio_dim)
            visual_feat: (B, visual_dim)
        Returns:
            logits: (B, num_classes)
        """
        t = self.text_proj(text_feat)     # (B, H)
        a = self.audio_proj(audio_feat)   # (B, H)
        v = self.visual_proj(visual_feat) # (B, H)

        # Stack modalities and compute attention weights
        stack = torch.stack([t, a, v], dim=1)  # (B, 3, H)
        attn_w = F.softmax(self.fusion_attn(stack), dim=1)  # (B, 3, 1)
        fused = (stack * attn_w).sum(dim=1)  # (B, H)

        return self.classifier(fused)


class ClassificationLoss(nn.Module):
    """
    Combined loss: cross-entropy + KD (KL divergence from teacher soft labels).

    Paper (§3.3): MLLM distillation via soft label transfer.
    Formula:
        L = α * CE(logits, hard_labels) + (1-α) * KL(student_soft || teacher_soft)
    """

    def __init__(self, alpha=0.5, temperature=3.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.ce = nn.CrossEntropyLoss()

    def forward(self, student_logits, hard_labels, teacher_soft_labels=None):
        ce_loss = self.ce(student_logits, hard_labels)

        if teacher_soft_labels is None:
            return ce_loss

        # KL divergence with temperature scaling
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        kd_loss = F.kl_div(student_soft, teacher_soft_labels, reduction="batchmean")
        kd_loss = kd_loss * (self.temperature ** 2)

        return self.alpha * ce_loss + (1 - self.alpha) * kd_loss
