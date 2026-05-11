"""
AFMRL: Attribute-Enhanced Fine-Grained Multi-Modal Representation Learning in E-commerce
arXiv: 2604.20135

Faithful PyTorch reproduction of the two-stage framework:
  Stage 1 — AGCL: Attribute-Guided Contrastive Learning
  Stage 2 — RAR:  Retrieval-aware Attribute Reinforcement

From Taobao & Tmall Group, Alibaba.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Attribute Generator (MLLM-backed in production; stub with learnable MLP here)
# ─────────────────────────────────────────────────────────────────────────────

class AttributeGenerator(nn.Module):
    """
    In the paper, a MLLM (e.g., LLaVA-style model) extracts structured product
    attributes from image+text input (color, material, style, category, etc.).

    Here we implement a learnable projection as a faithful interface stub.
    Replace this with an actual MLLM in production.

    Input:  fused image-text representation (B, fusion_dim)
    Output: attribute embedding (B, attr_dim) + discrete attribute logits (B, num_attrs)
    """

    def __init__(self, fusion_dim: int, attr_dim: int, num_attrs: int = 64):
        super().__init__()
        self.num_attrs = num_attrs
        self.attr_proj = nn.Sequential(
            nn.Linear(fusion_dim, attr_dim * 2),
            nn.GELU(),
            nn.LayerNorm(attr_dim * 2),
            nn.Linear(attr_dim * 2, attr_dim),
        )
        # Discrete attribute prediction head (multi-label)
        self.attr_classifier = nn.Linear(attr_dim, num_attrs)

    def forward(self, fused: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attr_emb = self.attr_proj(fused)                          # (B, attr_dim)
        attr_logits = self.attr_classifier(attr_emb)              # (B, num_attrs)
        return attr_emb, attr_logits


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Modal Encoder (image + text fusion)
# ─────────────────────────────────────────────────────────────────────────────

class MultiModalEncoder(nn.Module):
    """
    Produces a fused image-text representation.

    In production: image_enc = CLIP/ViT image encoder,
                   text_enc  = CLIP/LLM text encoder.
    Here we use simple projection MLPs with the correct interface.

    Formula:
        v = MLP_v(image_feat)
        t = MLP_t(text_feat)
        fused = LayerNorm(v + t)           [additive fusion]
    """

    def __init__(self, image_dim: int, text_dim: int, fusion_dim: int):
        super().__init__()
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, fusion_dim),
            nn.GELU(),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, fusion_dim),
            nn.GELU(),
        )
        self.norm = nn.LayerNorm(fusion_dim)

    def forward(
        self, image_feat: torch.Tensor, text_feat: torch.Tensor
    ) -> torch.Tensor:
        v = self.image_proj(image_feat)
        t = self.text_proj(text_feat)
        return self.norm(v + t)


# ─────────────────────────────────────────────────────────────────────────────
# Representation Head (produces retrieval embeddings)
# ─────────────────────────────────────────────────────────────────────────────

class RepresentationHead(nn.Module):
    """
    Maps fused representation + attribute embedding → final L2-normalized
    retrieval embedding for contrastive learning.

    e = L2_norm( MLP( concat(fused, attr_emb) ) )
    """

    def __init__(self, fusion_dim: int, attr_dim: int, embed_dim: int):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(fusion_dim + attr_dim, embed_dim * 2),
            nn.GELU(),
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(
        self, fused: torch.Tensor, attr_emb: torch.Tensor
    ) -> torch.Tensor:
        combined = torch.cat([fused, attr_emb], dim=-1)
        emb = self.projector(combined)
        return F.normalize(emb, dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: AGCL — Attribute-Guided Contrastive Learning Loss
# ─────────────────────────────────────────────────────────────────────────────

class AGCLLoss(nn.Module):
    """
    Attribute-Guided Contrastive Learning (AGCL).

    Key ideas (paper Section 3.1):
    1. Hard Negative Mining: use attribute similarity to find semantically hard negatives
       (products that share few attributes with anchor are easier; products that share
       many attributes but differ are hard negatives).
    2. False Negative Filtering: if attribute overlap exceeds threshold tau_fn,
       the pair is likely a true positive — exclude from negative set.

    Loss:
        L_AGCL = -log( exp(sim(q, k+) / temp) /
                        sum_{k- in HN, not FN} exp(sim(q, k-) / temp) )

    where HN = hard negatives (top-m by attribute similarity),
          FN filter = pairs with attr_overlap > tau_fn are excluded.

    Args:
        temperature:  contrastive temperature tau
        tau_fn:       false negative filtering threshold (attribute overlap)
        top_hard_k:   number of hard negatives to select per anchor
    """

    def __init__(
        self,
        temperature: float = 0.07,
        tau_fn: float = 0.7,
        top_hard_k: int = 10,
    ):
        super().__init__()
        self.temperature = temperature
        self.tau_fn = tau_fn
        self.top_hard_k = top_hard_k

    def forward(
        self,
        query_emb: torch.Tensor,          # (B, D) — anchor queries
        pos_emb: torch.Tensor,            # (B, D) — positive pairs
        neg_embs: torch.Tensor,           # (B, N, D) — candidate negatives pool
        query_attr: torch.Tensor,         # (B, num_attrs) — query attribute logits
        neg_attrs: torch.Tensor,          # (B, N, num_attrs) — negative attribute logits
    ) -> torch.Tensor:
        B, N, D = neg_embs.shape

        # Compute attribute overlap (cosine sim on binary attribute vectors)
        q_attr_bin = torch.sigmoid(query_attr)                    # (B, A)
        n_attr_bin = torch.sigmoid(neg_attrs)                     # (B, N, A)
        attr_overlap = torch.bmm(
            q_attr_bin.unsqueeze(1), n_attr_bin.transpose(1, 2)  # (B, 1, N)
        ).squeeze(1) / (q_attr_bin.norm(dim=-1, keepdim=True) + 1e-8)

        # False Negative mask: exclude negatives with high attribute overlap
        fn_mask = attr_overlap > self.tau_fn                      # (B, N) bool

        # Hard Negative selection: top-k negatives by attribute overlap
        # (not too similar = FN, but more similar than random = HN)
        _, hard_idx = attr_overlap.topk(
            min(self.top_hard_k, N), dim=-1, largest=True
        )                                                          # (B, top_hard_k)

        # Positive similarity
        pos_sim = (query_emb * pos_emb).sum(dim=-1, keepdim=True) / self.temperature  # (B, 1)

        # Gather hard negatives
        hard_neg_embs = torch.gather(
            neg_embs,
            1,
            hard_idx.unsqueeze(-1).expand(-1, -1, D)
        )                                                          # (B, top_hard_k, D)
        hard_fn_mask = torch.gather(fn_mask.float(), 1, hard_idx) # (B, top_hard_k)

        neg_sims = torch.bmm(
            query_emb.unsqueeze(1), hard_neg_embs.transpose(1, 2)
        ).squeeze(1) / self.temperature                            # (B, top_hard_k)

        # Mask out false negatives from denominator
        neg_sims = neg_sims.masked_fill(hard_fn_mask.bool(), -1e9)

        # InfoNCE loss
        logits = torch.cat([pos_sim, neg_sims], dim=-1)           # (B, 1 + top_hard_k)
        labels = torch.zeros(B, dtype=torch.long, device=query_emb.device)
        loss = F.cross_entropy(logits, labels)
        return loss


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: RAR — Retrieval-aware Attribute Reinforcement Loss
# ─────────────────────────────────────────────────────────────────────────────

class RARLoss(nn.Module):
    """
    Retrieval-aware Attribute Reinforcement (RAR).

    Uses retrieval performance as a reward signal to train the attribute generator.
    The MLLM attribute generator receives a reward when its attributes improve
    retrieval ranking.

    Simplified formulation (faithful to paper intent):
        R(a) = retrieval_improvement(a) - baseline_retrieval
        L_RAR = -R(a) * log P(a | image, text)    [REINFORCE-style]

    Here we implement a differentiable proxy: the attribute embedding's alignment
    with the positive retrieval embedding serves as the reward signal.

    L_RAR = -cosine_sim(attr_emb, pos_emb_sg)  (pos_emb stop-gradient'd as reward)

    This trains the attribute generator to produce attribute embeddings that
    align with positive retrieval pairs — faithful to the paper's reward concept.
    """

    def __init__(self, reward_scale: float = 0.1):
        super().__init__()
        self.reward_scale = reward_scale

    def forward(
        self,
        attr_emb: torch.Tensor,   # (B, attr_dim) — from attribute generator
        pos_emb: torch.Tensor,    # (B, D)         — positive retrieval embedding
        attr_proj: nn.Module,     # linear to align attr_dim → D
    ) -> torch.Tensor:
        # Project attribute embedding to retrieval space
        attr_in_retrieval_space = F.normalize(attr_proj(attr_emb), dim=-1)
        # Stop-gradient on retrieval embedding (treat as reward signal)
        pos_emb_sg = pos_emb.detach()
        # Reward: how well attribute aligns with retrieval positive
        reward = (attr_in_retrieval_space * pos_emb_sg).sum(dim=-1)  # (B,)
        loss = -self.reward_scale * reward.mean()
        return loss


# ─────────────────────────────────────────────────────────────────────────────
# Full AFMRL Model
# ─────────────────────────────────────────────────────────────────────────────

class AFMRL(nn.Module):
    """
    AFMRL: Attribute-Enhanced Fine-Grained Multi-Modal Representation Learning

    Two-stage pipeline:
    ┌─────────────────────────────────────────────────────────┐
    │  Image feat + Text feat                                  │
    │         ↓                                               │
    │   MultiModalEncoder → fused_repr                        │
    │         ↓                                               │
    │   AttributeGenerator → attr_emb, attr_logits            │
    │         ↓                                               │
    │   RepresentationHead → final_emb (L2-normalized)        │
    │                                                         │
    │  Stage 1 Loss: AGCL (attr-guided contrastive)           │
    │  Stage 2 Loss: RAR  (retrieval-reward for attr gen)     │
    └─────────────────────────────────────────────────────────┘

    Args:
        image_dim:   image feature dimension (e.g., 768 for ViT-B/16)
        text_dim:    text feature dimension  (e.g., 768 for CLIP text)
        fusion_dim:  multimodal fusion dimension
        attr_dim:    attribute embedding dimension
        embed_dim:   final retrieval embedding dimension
        num_attrs:   number of discrete attribute categories
    """

    def __init__(
        self,
        image_dim: int = 768,
        text_dim: int = 768,
        fusion_dim: int = 512,
        attr_dim: int = 256,
        embed_dim: int = 256,
        num_attrs: int = 64,
    ):
        super().__init__()
        self.encoder = MultiModalEncoder(image_dim, text_dim, fusion_dim)
        self.attr_gen = AttributeGenerator(fusion_dim, attr_dim, num_attrs)
        self.repr_head = RepresentationHead(fusion_dim, attr_dim, embed_dim)
        # Projection for RAR: maps attr_dim → embed_dim for reward computation
        self.attr_to_embed = nn.Linear(attr_dim, embed_dim)

    def encode(
        self, image_feat: torch.Tensor, text_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            emb:          (B, embed_dim) L2-normalized retrieval embedding
            attr_emb:     (B, attr_dim)  attribute embedding
            attr_logits:  (B, num_attrs) discrete attribute predictions
        """
        fused = self.encoder(image_feat, text_feat)
        attr_emb, attr_logits = self.attr_gen(fused)
        emb = self.repr_head(fused, attr_emb)
        return emb, attr_emb, attr_logits

    def forward(
        self,
        query_image: torch.Tensor,
        query_text: torch.Tensor,
        pos_image: torch.Tensor,
        pos_text: torch.Tensor,
        neg_images: Optional[torch.Tensor] = None,
        neg_texts: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Full forward pass for training.

        Args:
            query_image:  (B, image_dim)
            query_text:   (B, text_dim)
            pos_image:    (B, image_dim)  positive pair image
            pos_text:     (B, text_dim)   positive pair text
            neg_images:   (B, N, image_dim) candidate negatives
            neg_texts:    (B, N, text_dim)

        Returns:
            dict with embeddings and attribute logits for loss computation
        """
        q_emb, q_attr_emb, q_attr_logits = self.encode(query_image, query_text)
        p_emb, p_attr_emb, p_attr_logits = self.encode(pos_image, pos_text)

        out = {
            "q_emb": q_emb,
            "p_emb": p_emb,
            "q_attr_emb": q_attr_emb,
            "q_attr_logits": q_attr_logits,
            "p_attr_emb": p_attr_emb,
            "attr_to_embed": self.attr_to_embed,
        }

        if neg_images is not None and neg_texts is not None:
            B, N, _ = neg_images.shape
            n_img_flat = neg_images.reshape(B * N, -1)
            n_txt_flat = neg_texts.reshape(B * N, -1)
            n_emb, _, n_attr_logits = self.encode(n_img_flat, n_txt_flat)
            out["neg_embs"] = n_emb.reshape(B, N, -1)
            out["neg_attr_logits"] = n_attr_logits.reshape(B, N, -1)

        return out
