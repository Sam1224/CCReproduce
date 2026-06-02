"""
MLLM Teacher for HybridMod knowledge distillation.

In the paper, a large multimodal LLM (e.g. GPT-4V or an internal MLLM)
acts as teacher for both pipelines:
  - Pipeline A: generates soft labels / logits for classification distillation
  - Pipeline B: provides high-quality embeddings to seed the reference gallery

This module provides a stub teacher interface with a lightweight surrogate
(two-layer MLP) that mimics the teacher's output shape. In practice, replace
`MLLMTeacherSurrogate` with calls to a real MLLM API / local model.

Distillation loss (KL divergence):
    L_distill = KL(sigma(z_student / T) || sigma(z_teacher / T))   (Eq. in paper)

where T is the temperature.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLLMTeacherSurrogate(nn.Module):
    """
    Lightweight surrogate for the MLLM teacher (for reproducibility without access to MLLM).
    Replace with a real MLLM (e.g. LLaVA, InternVL, GPT-4V) in production.
    """

    def __init__(self, text_dim: int, visual_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(text_dim + visual_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.head = nn.Linear(hidden_dim, num_classes)

    @torch.no_grad()
    def forward(self, text_feat, visual_feat):
        combined = torch.cat([text_feat, visual_feat], dim=-1)
        h = self.fusion(combined)
        logits = self.head(h)
        return logits, h  # (B, C), (B, H)


class KnowledgeDistillationLoss(nn.Module):
    """
    Combines task loss (cross-entropy) with distillation loss (KL divergence).

    L_total = (1 - alpha) * L_CE + alpha * T^2 * L_KD

    Args:
        temperature: Distillation temperature T. Higher T → softer teacher distribution.
        alpha: Weight for distillation loss vs. hard-label CE loss.
    """

    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        super().__init__()
        self.T = temperature
        self.alpha = alpha

    def forward(self, student_logits, teacher_logits, hard_labels):
        # Hard-label cross-entropy
        ce_loss = F.cross_entropy(student_logits, hard_labels)

        # Soft-label KL divergence
        student_soft = F.log_softmax(student_logits / self.T, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.T, dim=-1)
        kd_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (self.T ** 2)

        return (1 - self.alpha) * ce_loss + self.alpha * kd_loss, ce_loss, kd_loss


def build_reference_gallery(teacher: MLLMTeacherSurrogate, dataloader, model, device):
    """
    Pre-compute MLLM teacher embeddings for reference samples and populate gallery.
    Called once before training the similarity pipeline.
    """
    teacher.eval()
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            text_feat = batch["text_feat"].to(device)
            audio_feat = batch["audio_feat"].to(device)
            visual_feat = batch["visual_feat"].to(device)
            labels = batch["label"].to(device)

            # Teacher provides high-quality embeddings
            _, teacher_emb = teacher(text_feat, visual_feat)

            # Update reference gallery with teacher embeddings
            model.similarity_pipeline.gallery.update(teacher_emb.cpu(), labels.cpu())

    print(f"Gallery populated with {model.similarity_pipeline.gallery.gallery_size} entries.")
