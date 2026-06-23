"""
Atomic Intent Unit (AIU) Extractor.

Core idea from AIR: instead of feeding the full noisy behavior sequence to the LLM,
first compress it into a small set of high-signal atomic intent statements.

The extractor performs two operations:
1. Signal filtering: identify which interactions carry true purchase intent
   (purchases > follows > likes > watches in signal value)
2. Semantic compression: produce 1-3 natural-language AIU sentences

In production AIR: a dedicated lightweight encoder trained with contrastive
learning on (behavior_sequence, purchase_outcome) pairs extracts AIUs.
Here we implement a rule-based extractor as a structural proxy.
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import Optional


# Signal weights: how much each interaction type reveals purchase intent
INTENT_SIGNAL_WEIGHTS = {
    "purchased": 1.0,
    "followed_creator": 0.6,
    "liked": 0.3,
    "watched": 0.1,
}


def score_content_interaction(interaction: dict) -> float:
    """Score a single content interaction for purchase intent signal."""
    score = 0.0
    score += interaction.get("watched", 0) * INTENT_SIGNAL_WEIGHTS["watched"]
    score += interaction.get("liked", 0) * INTENT_SIGNAL_WEIGHTS["liked"]
    score += interaction.get("followed_creator", 0) * INTENT_SIGNAL_WEIGHTS["followed_creator"]
    return score


def extract_high_signal_interactions(user: dict, top_k: int = 8) -> list[dict]:
    """
    Filter the content history to the top-K highest intent-signal interactions.
    This is the key noise-reduction step of AIR.
    """
    content_hist = user.get("content_history", [])
    scored = [(h, score_content_interaction(h)) for h in content_hist]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [h for h, s in scored[:top_k] if s > 0.0]


def build_aiu_text(user: dict, high_signal: list[dict]) -> str:
    """
    Build the compressed AIU text from high-signal interactions and purchase history.
    Format follows AIR's atomic intent structure: concise, purchase-intent-focused.
    """
    purchase_hist = user.get("purchase_history", [])
    preferred_content = user.get("preferred_content_cats", [])
    preferred_products = user.get("preferred_product_cats", [])

    # Aggregate content signals
    content_cats = list({h["category"] for h in high_signal})
    liked_creators = [h for h in high_signal if h.get("followed_creator")]
    purchased_cats = list({p["category"] for p in purchase_hist})

    aiu_parts = []

    if content_cats:
        aiu_parts.append(
            f"User's high-engagement content: {', '.join(content_cats[:3])}."
        )
    if purchased_cats:
        aiu_parts.append(
            f"Cross-domain purchase intent: {', '.join(purchased_cats[:3])} products."
        )
    if liked_creators:
        creator_tiers = list({h["creator_tier"] for h in liked_creators})
        aiu_parts.append(
            f"Follows {creator_tiers[0]} creators — high purchase conversion potential."
        )

    return " ".join(aiu_parts) if aiu_parts else "Insufficient behavioral signal."


class AIUExtractorRuleBased:
    """
    Rule-based AIU extractor (toy implementation).
    Mimics the production AIU extraction without model inference.
    """

    def __init__(self, top_k: int = 8):
        self.top_k = top_k

    def extract(self, user: dict) -> str:
        high_signal = extract_high_signal_interactions(user, self.top_k)
        return build_aiu_text(user, high_signal)

    def extract_batch(self, users: list[dict]) -> list[str]:
        return [self.extract(u) for u in users]


class AIUExtractorNeural(nn.Module):
    """
    Neural AIU extractor (lightweight encoder + attention-based compression).

    Architecture:
    - Embed each content interaction as a feature vector
    - Self-attention over the sequence to weight by intent relevance
    - Pool → linear → AIU embedding (used in lieu of text for training efficiency)
    """

    def __init__(
        self,
        n_content_cats: int = 10,
        n_creator_tiers: int = 4,
        hidden_dim: int = 128,
        aiu_dim: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.aiu_dim = aiu_dim

        # Categorical embeddings for content features
        self.cat_embed = nn.Embedding(n_content_cats + 1, 32)      # +1 for padding
        self.tier_embed = nn.Embedding(n_creator_tiers + 1, 16)

        # Interaction signal embedding: [watched, liked, followed] as scalars
        self.signal_proj = nn.Linear(3, 16)

        # Combine per-interaction features
        self.item_proj = nn.Linear(32 + 16 + 16, hidden_dim)

        # Intent-relevance attention (captures which interactions matter for purchasing)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=True,
        )
        self.intent_attn = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Compress sequence → AIU embedding
        self.aiu_proj = nn.Linear(hidden_dim, aiu_dim)

    def encode_interaction(
        self,
        cat_ids: torch.Tensor,    # [B, T]
        tier_ids: torch.Tensor,   # [B, T]
        signals: torch.Tensor,    # [B, T, 3] (watched, liked, followed)
    ) -> torch.Tensor:
        cat_emb = self.cat_embed(cat_ids)        # [B, T, 32]
        tier_emb = self.tier_embed(tier_ids)     # [B, T, 16]
        sig_emb = self.signal_proj(signals)      # [B, T, 16]

        combined = torch.cat([cat_emb, tier_emb, sig_emb], dim=-1)  # [B, T, 64]
        return self.item_proj(combined)  # [B, T, hidden_dim]

    def forward(
        self,
        cat_ids: torch.Tensor,    # [B, T]
        tier_ids: torch.Tensor,   # [B, T]
        signals: torch.Tensor,    # [B, T, 3]
        padding_mask: Optional[torch.Tensor] = None,  # [B, T] True = pad
    ) -> torch.Tensor:
        """
        Returns AIU embeddings: [B, aiu_dim]
        """
        item_feats = self.encode_interaction(cat_ids, tier_ids, signals)  # [B, T, H]

        # Transformer with intent-relevance attention
        attended = self.intent_attn(item_feats, src_key_padding_mask=padding_mask)  # [B, T, H]

        # Mean-pool (mask padded positions)
        if padding_mask is not None:
            mask_float = (~padding_mask).float().unsqueeze(-1)  # [B, T, 1]
            pooled = (attended * mask_float).sum(dim=1) / (mask_float.sum(dim=1) + 1e-8)
        else:
            pooled = attended.mean(dim=1)  # [B, H]

        return self.aiu_proj(pooled)  # [B, aiu_dim]
