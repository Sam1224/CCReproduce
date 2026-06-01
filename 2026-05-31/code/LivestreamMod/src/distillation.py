"""
MLLM teacher distillation for the similarity path.

In the paper, a large MLLM (e.g., InternVL) serves as teacher, generating
soft labels (probability distributions over violation categories) for each
livestream segment. The similarity re-ranker is distilled from these soft labels.

In this toy implementation, we use a larger MLP as the "teacher" model.

Loss = KL(teacher_soft || student_soft) + CE(student, hard_label)
     = distillation loss + cross-entropy loss

Reference: paper Section 3.2 "MLLM-Boosted Similarity Matching via Distillation"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .models import ClassificationPath


class TeacherMLP(nn.Module):
    """Simulated MLLM teacher: a larger MLP with pretrained/fixed weights."""
    def __init__(self, input_dim: int = 384, num_classes: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, num_classes),
        )

    @torch.no_grad()
    def soft_labels(self, fused: torch.Tensor, temperature: float = 3.0):
        """Generate soft labels with temperature scaling."""
        logits = self.net(fused)
        return F.softmax(logits / temperature, dim=-1)


class DistillationLoss(nn.Module):
    """
    Combined distillation + task loss.
    L = λ * KL(teacher || student) + (1-λ) * CE(student, hard_label)

    Formula (Eq. 3 in paper, simplified):
    L_dist = KL(p_teacher(y|x) || p_student(y|x))
    L_task = -log p_student(y_true | x)
    L_total = λ * L_dist + (1-λ) * L_task
    """
    def __init__(self, temperature: float = 3.0, lam: float = 0.7):
        super().__init__()
        self.T = temperature
        self.lam = lam

    def forward(self, student_logits: torch.Tensor,
                teacher_soft: torch.Tensor,
                hard_labels: torch.Tensor) -> torch.Tensor:
        # Distillation: KL divergence between teacher soft and student soft
        student_soft = F.log_softmax(student_logits / self.T, dim=-1)
        kl_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (self.T ** 2)

        # Task: cross-entropy with hard labels
        ce_loss = F.cross_entropy(student_logits, hard_labels)

        return self.lam * kl_loss + (1 - self.lam) * ce_loss


def pretrain_teacher(teacher: TeacherMLP, train_loader, epochs: int = 10,
                     device: str = "cpu") -> TeacherMLP:
    """Pre-train teacher on labeled data so it generates meaningful soft labels."""
    teacher = teacher.to(device)
    opt = torch.optim.Adam(teacher.parameters(), lr=1e-3)
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in train_loader:
            fused = batch["fused"].to(device)
            labels = batch["label"].to(device)
            logits = teacher.net(fused)
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"  Teacher pretrain epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}")
    return teacher
