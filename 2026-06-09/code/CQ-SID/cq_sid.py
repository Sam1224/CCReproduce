"""
CQ-SID: Category-and-Query Constrained Semantic ID for E-commerce Generative Retrieval.

From paper: "Efficient Generative Retrieval for E-commerce Search with
Semantic Cluster IDs and Expert-Guided RL" (arXiv:2605.14434)

Key components:
1. Category-aware contrastive learning: products in same category cluster together
2. Query-item contrastive learning: products matching a query cluster together
3. RQ-VAE: Residual Quantized VAE for hierarchical semantic IDs

Paper equations:
    z_p = Encoder(product_text, category)        # product embedding
    z_q = QueryEncoder(query_text)                # query embedding

    L_cat = CategoryContrastive(z_p)             # pull same-category products
    L_qi  = QueryItemContrastive(z_p, z_q)       # pull query-matched products
    L_rq  = RQVAELoss(z_p)                       # quantization loss

    CQ-SID = RQ(z_p)  (K levels of hierarchical discrete codes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size: int, dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.codebook = nn.Embedding(codebook_size, dim)
        self.commitment_cost = commitment_cost
        nn.init.uniform_(self.codebook.weight, -1 / codebook_size, 1 / codebook_size)

    def forward(self, z):
        dist = (
            z.pow(2).sum(-1, keepdim=True)
            - 2 * z @ self.codebook.weight.T
            + self.codebook.weight.pow(2).sum(-1)
        )
        codes = dist.argmin(-1)
        z_q = self.codebook(codes)
        z_q_st = z + (z_q - z).detach()
        loss = F.mse_loss(z_q, z.detach()) + self.commitment_cost * F.mse_loss(z, z_q.detach())
        residual = z - z_q.detach()
        return z_q_st, codes, residual, loss


class ResidualVQ(nn.Module):
    def __init__(self, n_levels: int, codebook_size: int, dim: int):
        super().__init__()
        self.vqs = nn.ModuleList([VectorQuantizer(codebook_size, dim) for _ in range(n_levels)])
        self.n_levels = n_levels

    def forward(self, z):
        residual = z
        all_codes, total_loss = [], 0.0
        z_q_sum = torch.zeros_like(z)
        for vq in self.vqs:
            z_q, codes, residual, loss = vq(residual)
            all_codes.append(codes)
            total_loss += loss
            z_q_sum = z_q_sum + z_q
        return torch.stack(all_codes, dim=1), total_loss, z_q_sum  # (B, K), scalar, (B, D)


class CQSIDEncoder(nn.Module):
    """
    Product encoder with category and query constraints.

    Produces CQ-SID hierarchical codes for products.
    """

    def __init__(
        self,
        feat_dim: int = 128,
        n_categories: int = 50,
        cat_embed_dim: int = 32,
        hidden_dim: int = 128,
        n_levels: int = 4,
        codebook_size: int = 256,
    ):
        super().__init__()
        self.cat_embed = nn.Embedding(n_categories, cat_embed_dim)
        self.product_encoder = nn.Sequential(
            nn.Linear(feat_dim + cat_embed_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.query_encoder = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.rq = ResidualVQ(n_levels, codebook_size, hidden_dim)
        self.hidden_dim = hidden_dim

    def encode_product(self, product_feat, product_cat):
        cat_e = self.cat_embed(product_cat)
        return F.normalize(self.product_encoder(torch.cat([product_feat, cat_e], -1)), dim=-1)

    def encode_query(self, query_feat):
        return F.normalize(self.query_encoder(query_feat), dim=-1)

    def quantize(self, z):
        return self.rq(z)

    def forward(self, product_feat, product_cat, query_feat=None):
        """
        Returns codes, vq_loss, and optionally contrastive losses.
        """
        z_p = self.encode_product(product_feat, product_cat)
        codes, vq_loss, z_q = self.quantize(z_p)

        cat_loss = torch.tensor(0.0, device=product_feat.device)
        qi_loss = torch.tensor(0.0, device=product_feat.device)

        if query_feat is not None:
            z_q_enc = self.encode_query(query_feat)
            # Query-Item contrastive loss (InfoNCE)
            # Positive: query should be close to matching product
            qi_loss = info_nce_loss(z_q_enc, z_p)

        return codes, vq_loss, cat_loss, qi_loss, z_p


def info_nce_loss(anchors: torch.Tensor, positives: torch.Tensor, temperature: float = 0.07):
    """
    InfoNCE / NT-Xent contrastive loss.
    anchors: (B, D), positives: (B, D)
    Each anchor's positive is the corresponding positive in the batch.
    """
    B = anchors.size(0)
    sim_matrix = anchors @ positives.T / temperature    # (B, B)
    labels = torch.arange(B, device=anchors.device)
    return F.cross_entropy(sim_matrix, labels)
