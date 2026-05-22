"""Contrastive loss for TGQ-Former I2I retrieval training."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """In-batch InfoNCE + triplet margin loss.

    In-batch InfoNCE (SimCLR-style):
        L_nce = -log( exp(sim(q,p)/τ) / Σ_k exp(sim(q,k)/τ) )

    Triplet margin (margin-based):
        L_tri = max(0, sim(q,neg) - sim(q,pos) + margin)

    Combined: L = λ_nce * L_nce + λ_tri * L_tri
    """

    def __init__(self, temperature: float = 0.07, margin: float = 0.2,
                 lam_nce: float = 1.0, lam_tri: float = 0.5):
        super().__init__()
        self.tau = temperature
        self.margin = margin
        self.lam_nce = lam_nce
        self.lam_tri = lam_tri

    def forward(self, q: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor) -> dict:
        """
        q, pos, neg: (B, emb_dim) L2-normalized

        Returns dict with total loss and components.
        """
        B = q.size(0)

        # --- In-batch InfoNCE ---
        # Similarity matrix: (B, 2B) — positives + in-batch negatives
        all_keys = torch.cat([pos, neg], dim=0)           # (2B, d)
        sim = torch.matmul(q, all_keys.T) / self.tau       # (B, 2B)
        labels = torch.arange(B, device=q.device)          # positives at index 0..B-1
        loss_nce = F.cross_entropy(sim, labels)

        # --- Triplet margin ---
        sim_pos = (q * pos).sum(-1)   # (B,)
        sim_neg = (q * neg).sum(-1)   # (B,)
        loss_tri = F.relu(sim_neg - sim_pos + self.margin).mean()

        loss = self.lam_nce * loss_nce + self.lam_tri * loss_tri
        return {"loss": loss, "nce": loss_nce.item(), "tri": loss_tri.item()}
