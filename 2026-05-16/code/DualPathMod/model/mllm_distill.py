"""
MLLM Knowledge Distillation for Dual-Path Moderation (arXiv 2512.03553 §3.2)

The paper's key insight: MLLM (teacher) is too slow for real-time inference
but provides rich understanding. We distill its knowledge into:
  1. Soft classification labels for Path 1 (classification path)
  2. Semantic embedding targets for Path 2 (similarity path)

The distilled lightweight model (Student) handles real-time inference,
while the MLLM only runs offline for data annotation and distillation.

Paper §3.2: "Both pipelines are boosted by MLLM knowledge distillation.
The classification pipeline's soft labels are provided by the MLLM.
The similarity pipeline's embedding space is aligned with MLLM representations."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class MockMLLMTeacher(nn.Module):
    """
    Mock MLLM teacher model for distillation.
    In production: large MLLM (e.g., Qwen-VL-72B or LLaVA-1.6-34B) run offline.

    Produces:
      - Soft classification probabilities over violation types
      - Rich semantic embeddings from penultimate layer
    """

    def __init__(self, num_classes: int = 7, embed_dim: int = 128, teacher_hidden: int = 1024):
        super().__init__()
        # Simulate a larger model's output head
        self.hidden = nn.Sequential(
            nn.Linear(1024, teacher_hidden),
            nn.GELU(),
            nn.Linear(teacher_hidden, teacher_hidden),
            nn.GELU(),
        )
        self.class_head = nn.Linear(teacher_hidden, num_classes)
        self.embed_head = nn.Sequential(
            nn.Linear(teacher_hidden, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        # MLLM always runs at higher temperature (more calibrated distributions)
        self.temperature = 2.0

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          soft_labels: [B, num_classes] softmax probabilities at MLLM temperature
          teacher_embeds: [B, embed_dim] L2-normalized semantic embeddings
        """
        # Pad/project features to teacher's input dim
        if features.size(-1) != 1024:
            features = F.pad(features, (0, 1024 - features.size(-1)))

        h = self.hidden(features)
        class_logits = self.class_head(h)
        soft_labels = F.softmax(class_logits / self.temperature, dim=-1)

        teacher_embeds = self.embed_head(h)
        teacher_embeds = F.normalize(teacher_embeds, dim=-1)

        return soft_labels, teacher_embeds


class DistillationLoss(nn.Module):
    """
    Combined distillation loss for both paths.

    Path 1 distillation (classification):
      L_kl = KL(p_student / T || p_teacher / T) * T²   [Hinton et al. 2015]

    Path 2 distillation (embeddings):
      L_embed = 1 - cosine_similarity(student_embed, teacher_embed)
              = L2(student_embed - teacher_embed)²  [since both L2-normalized]

    Total: L_distill = α * L_kl + β * L_embed
    """

    def __init__(self, temperature: float = 4.0, alpha: float = 0.5, beta: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta

    def path1_kl_loss(
        self,
        student_logits: torch.Tensor,
        teacher_soft_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        KL divergence from student logits to teacher soft labels.
        Formula (Hinton 2015):
          L_KL = T² * KL(student_soft || teacher_soft)
               = T² * Σ teacher_soft * log(teacher_soft / student_soft)
        """
        student_log_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = teacher_soft_labels  # already softmaxed at teacher temperature
        # Re-normalize teacher at current temperature
        teacher_log = torch.log(teacher_soft + 1e-8)
        teacher_at_T = F.softmax(teacher_log * (self.temperature / 2.0), dim=-1)
        kl = F.kl_div(student_log_soft, teacher_at_T, reduction="batchmean")
        return kl * (self.temperature ** 2)

    def path2_embed_loss(
        self,
        student_embeds: torch.Tensor,
        teacher_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Cosine embedding alignment loss.
        Formula: L_embed = 1 - (e_s · e_t)  where e_s, e_t are L2-normalized
        """
        cosine_sim = (student_embeds * teacher_embeds).sum(dim=-1)
        return (1 - cosine_sim).mean()

    def forward(
        self,
        student_logits: torch.Tensor,
        student_embeds: torch.Tensor,
        teacher_soft_labels: torch.Tensor,
        teacher_embeds: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        kl_loss = self.path1_kl_loss(student_logits, teacher_soft_labels)
        embed_loss = self.path2_embed_loss(student_embeds, teacher_embeds)
        total = self.alpha * kl_loss + self.beta * embed_loss
        return {
            "distill_loss": total,
            "kl_loss": kl_loss,
            "embed_align_loss": embed_loss,
        }


class OfflineMLLMAnnotator:
    """
    Offline MLLM annotation pipeline.
    Runs the teacher MLLM on a corpus of livestream clips to generate:
      1. Soft violation labels for Path 1 distillation
      2. Semantic embeddings for Path 2 distillation + violation store building

    In production:
      - Runs on GPU cluster with MLLM (e.g., 70B+ parameters)
      - Takes ~24 hours for a 1M-clip corpus
      - Output cached and used for student training
    """

    def __init__(self, teacher: MockMLLMTeacher, batch_size: int = 32):
        self.teacher = teacher
        self.batch_size = batch_size

    @torch.no_grad()
    def annotate_batch(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Annotate a batch of features with MLLM soft labels + embeddings."""
        self.teacher.eval()
        soft_labels, embeds = self.teacher(features)
        return {
            "soft_labels": soft_labels,
            "teacher_embeds": embeds,
        }

    @torch.no_grad()
    def annotate_dataset(self, all_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Annotate a full dataset in batches."""
        all_soft_labels, all_embeds = [], []
        for i in range(0, len(all_features), self.batch_size):
            batch = all_features[i:i + self.batch_size]
            annotations = self.annotate_batch(batch)
            all_soft_labels.append(annotations["soft_labels"])
            all_embeds.append(annotations["teacher_embeds"])
        return {
            "soft_labels": torch.cat(all_soft_labels, dim=0),
            "teacher_embeds": torch.cat(all_embeds, dim=0),
        }
