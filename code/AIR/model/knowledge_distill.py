"""
Online Knowledge Distillation module for AIR.

The LLM reasoner (teacher) runs offline and produces high-quality cross-domain
intent embeddings. At serving time, a lightweight student model must:
1. Accept only real-time signals (no LLM inference latency)
2. Produce embeddings close to the teacher's offline embeddings
3. Support O(1ms) inference for online ranking

Distillation approach:
- Teacher: AIRLLMReasoner (LLM → 256-dim intent embedding)
- Student: lightweight MLP over the neural AIU extractor's output
- Loss: MSE(student_emb, teacher_emb) + cross-entropy on product category prediction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AIRStudentModel(nn.Module):
    """
    Lightweight student model for online inference.

    Input: neural AIU embedding (from AIUExtractorNeural, computed at serving time
           from the last N minutes of user interactions)
    Output: cross-domain intent embedding (same dim as teacher's 256-dim output)

    In production: the AIU extractor runs continuously on streaming interactions;
    the student takes the latest AIU embedding and predicts intent in <1ms.
    """

    def __init__(
        self,
        aiu_dim: int = 256,         # input: AIU embedding from neural extractor
        intent_dim: int = 256,      # output: must match teacher's intent_dim
        hidden_dims: list = None,
        n_product_cats: int = 10,   # auxiliary head: predict product category
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]

        layers = []
        in_dim = aiu_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim

        self.backbone = nn.Sequential(*layers)
        self.intent_head = nn.Linear(in_dim, intent_dim)
        self.category_head = nn.Linear(in_dim, n_product_cats)  # auxiliary task

    def forward(self, aiu_embeddings: torch.Tensor) -> dict:
        """
        Args:
            aiu_embeddings: [B, aiu_dim] from neural AIU extractor

        Returns:
            dict with 'intent_embeddings' [B, intent_dim], 'cat_logits' [B, n_cats]
        """
        feat = self.backbone(aiu_embeddings)  # [B, hidden_dim]
        intent_emb = self.intent_head(feat)   # [B, intent_dim]
        cat_logits = self.category_head(feat) # [B, n_cats]
        return {
            "intent_embeddings": intent_emb,
            "cat_logits": cat_logits,
        }


class KnowledgeDistillationTrainer(nn.Module):
    """
    Manages the distillation training between teacher LLM and student model.

    Loss components:
    1. Distillation loss: MSE between student and teacher embeddings (main signal)
    2. Category prediction loss: cross-entropy on product category (auxiliary, keeps student grounded)
    3. Contrastive loss: positive pairs (same user's teacher and student) pulled together,
       negative pairs (different users) pushed apart
    """

    def __init__(
        self,
        student: AIRStudentModel,
        distill_weight: float = 1.0,
        category_weight: float = 0.3,
        contrastive_weight: float = 0.2,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.student = student
        self.distill_weight = distill_weight
        self.category_weight = category_weight
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature

    def contrastive_loss(
        self,
        student_emb: torch.Tensor,   # [B, D]
        teacher_emb: torch.Tensor,   # [B, D]
    ) -> torch.Tensor:
        """
        InfoNCE-style contrastive loss.
        Positive: (student_i, teacher_i) same user.
        Negative: (student_i, teacher_j) different users.
        """
        B = student_emb.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=student_emb.device)

        s_norm = F.normalize(student_emb, dim=-1)   # [B, D]
        t_norm = F.normalize(teacher_emb, dim=-1)   # [B, D]

        # Similarity matrix [B, B]
        sim = torch.matmul(s_norm, t_norm.T) / self.temperature
        labels = torch.arange(B, device=sim.device)
        return F.cross_entropy(sim, labels)

    def forward(
        self,
        aiu_embeddings: torch.Tensor,       # [B, aiu_dim] from neural extractor
        teacher_intent: torch.Tensor,       # [B, intent_dim] from offline LLM teacher
        product_cat_labels: Optional[torch.Tensor] = None,  # [B] int category labels
    ) -> dict:
        student_out = self.student(aiu_embeddings)
        student_intent = student_out["intent_embeddings"]  # [B, intent_dim]
        cat_logits = student_out["cat_logits"]             # [B, n_cats]

        # 1. Distillation loss: L2 distance to teacher
        loss_distill = F.mse_loss(student_intent, teacher_intent.detach())

        # 2. Contrastive loss
        loss_contrast = self.contrastive_loss(student_intent, teacher_intent.detach())

        # 3. Auxiliary category loss
        loss_cat = torch.tensor(0.0, device=aiu_embeddings.device)
        if product_cat_labels is not None:
            loss_cat = F.cross_entropy(cat_logits, product_cat_labels)

        total_loss = (
            self.distill_weight * loss_distill
            + self.contrastive_weight * loss_contrast
            + self.category_weight * loss_cat
        )

        return {
            "loss": total_loss,
            "loss_distill": loss_distill.item(),
            "loss_contrastive": loss_contrast.item(),
            "loss_category": loss_cat.item(),
            "student_intent_embeddings": student_intent,
        }


class AIROnlineRanker(nn.Module):
    """
    Online ranker that consumes student intent embeddings for CTR/CVR prediction.
    Same interface as Taiji's TaijiOnlineRanker for consistency.
    """

    def __init__(self, intent_dim: int = 256, product_dim: int = 64):
        super().__init__()
        self.intent_proj = nn.Linear(intent_dim, 128)
        self.product_proj = nn.Linear(product_dim, 128)
        self.mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.ctr_head = nn.Linear(128, 1)
        self.cvr_head = nn.Linear(128, 1)

    def forward(
        self,
        intent_embeddings: torch.Tensor,   # [B, intent_dim]
        product_embeddings: torch.Tensor,  # [B, product_dim]
    ) -> dict:
        intent_feat = self.intent_proj(intent_embeddings)   # [B, 128]
        prod_feat = self.product_proj(product_embeddings)   # [B, 128]

        combined = torch.cat([intent_feat, prod_feat], dim=-1)  # [B, 256]
        shared = self.mlp(combined)                              # [B, 128]

        ctr = torch.sigmoid(self.ctr_head(shared)).squeeze(-1)  # [B]
        cvr = torch.sigmoid(self.cvr_head(shared)).squeeze(-1)  # [B]
        return {"ctr": ctr, "cvr": cvr}
