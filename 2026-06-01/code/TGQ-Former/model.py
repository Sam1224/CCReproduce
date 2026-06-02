"""
TGQ-Former: Text-Guided Q-Former for Robust Multimodal E-Commerce Recommendation
Reproduction of arXiv: 2605.17366 (Tsinghua + JD.COM)

Core architecture:
  1. Frozen vision encoder → visual tokens V
  2. Text encoder → metadata embedding M
  3. Hybrid-Query Connector:
       a. Metadata-Anchored Query: cross-attention V×M → anchored features F_a
       b. Exploratory Query: self-attention V → exploratory features F_e
  4. Reliability-Aware Dual-Gated Vector Modulation:
       gate_a, gate_e = sigmoid(MLP(concat(F_a, F_e, noise_signal)))
       output = gate_a * F_a + gate_e * F_e
  5. Final visual embedding aligned with text for contrastive I2I retrieval

Paper Eq. (key formulas, approximated):
  F_a = CrossAttn(Q, K=V, V=V, guidance=M)    (text-anchored queries)
  F_e = SelfAttn(Q, K=V, V=V)                 (exploratory queries)
  gate_a = sigma(W_a [F_a; F_e; noise_est])
  gate_e = 1 - gate_a   (or independent gates)
  z_visual = gate_a * F_a + gate_e * F_e
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MetadataEncoder(nn.Module):
    """
    Encodes product metadata (title, category, attributes) into dense embeddings.
    In production: use a frozen BERT/RoBERTa. Here: learnable embedding.
    """

    def __init__(self, vocab_size: int = 30522, embed_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=4, dim_feedforward=embed_dim * 4,
            dropout=0.1, batch_first=True, norm_first=True
        )
        self.proj = nn.Linear(embed_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, token_ids, attention_mask=None):
        """
        Args:
            token_ids:      (B, L)
            attention_mask: (B, L) — 1 for valid tokens
        Returns:
            metadata_emb: (B, hidden_dim)
        """
        x = self.embed(token_ids)  # (B, L, embed_dim)
        mask = None if attention_mask is None else (attention_mask == 0)
        x = self.encoder(x, src_key_padding_mask=mask)
        # Mean pooling over valid tokens
        if attention_mask is not None:
            valid = attention_mask.float().unsqueeze(-1)
            x = (x * valid).sum(1) / valid.sum(1).clamp(min=1)
        else:
            x = x.mean(1)
        return self.norm(self.proj(x))


class MetadataAnchoredQuery(nn.Module):
    """
    Metadata-anchored stream of the Hybrid-Query Connector.

    Standard Q-Former learnable queries are conditioned on metadata M:
    the metadata embedding M is projected as key/value for cross-attention
    with visual tokens V as additional key/value.

    F_a = CrossAttn(Q + bias(M), K=[V; M_proj], V=[V; M_proj])

    This steers queries towards metadata-relevant visual regions,
    suppressing noise overlays (promotional stickers, cluttered backgrounds).
    """

    def __init__(self, num_queries: int, hidden_dim: int, num_visual_tokens: int, num_heads: int = 4):
        super().__init__()
        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        # Metadata conditions the queries via additive bias
        self.meta_bias_proj = nn.Linear(hidden_dim, hidden_dim)
        # Cross-attention to visual tokens
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, visual_tokens, metadata_emb):
        """
        Args:
            visual_tokens: (B, N_v, hidden_dim) — from vision encoder
            metadata_emb:  (B, hidden_dim) — from MetadataEncoder
        Returns:
            F_a: (B, num_queries, hidden_dim)
        """
        B = visual_tokens.size(0)
        idx = torch.arange(self.num_queries, device=visual_tokens.device)
        queries = self.query_embed(idx).unsqueeze(0).expand(B, -1, -1)  # (B, Q, H)
        # Add metadata as additive guidance bias
        meta_bias = self.meta_bias_proj(metadata_emb).unsqueeze(1)  # (B, 1, H)
        queries = queries + meta_bias
        # Cross-attend to visual tokens
        out, _ = self.cross_attn(queries, visual_tokens, visual_tokens)
        out = self.norm(queries + out)
        out = self.norm(out + self.ffn(out))
        return out  # (B, Q, H)


class ExploratoryQuery(nn.Module):
    """
    Exploratory stream of the Hybrid-Query Connector.

    Free queries attend to visual tokens without metadata guidance,
    capturing visual details not described in metadata (texture, color, style).

    F_e = SelfAttn(Q, K=V, V=V)   (no metadata conditioning)
    """

    def __init__(self, num_queries: int, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, visual_tokens):
        """
        Args:
            visual_tokens: (B, N_v, hidden_dim)
        Returns:
            F_e: (B, num_queries, hidden_dim)
        """
        B = visual_tokens.size(0)
        idx = torch.arange(self.num_queries, device=visual_tokens.device)
        queries = self.query_embed(idx).unsqueeze(0).expand(B, -1, -1)
        out, _ = self.cross_attn(queries, visual_tokens, visual_tokens)
        out = self.norm(queries + out)
        out = self.norm(out + self.ffn(out))
        return out  # (B, Q, H)


class ReliabilityAwareDualGatedModulation(nn.Module):
    """
    Reliability-Aware Dual-Gated Vector Modulation.

    Estimates the reliability of the metadata-anchored stream F_a
    (which degrades when product images are noisy/clutter-heavy).
    Adaptively gates the contribution of each stream.

    gate_a = sigma(MLP([F_a_pooled; F_e_pooled; noise_est]))
    gate_e = sigma(MLP([F_a_pooled; F_e_pooled; noise_est]))   (independent gate)
    z = gate_a * pool(F_a) + gate_e * pool(F_e)

    The noise_est is a learnable image quality signal (here: L2 norm of F_a - F_e
    as a proxy for disagreement between the two streams, per paper §3.3).
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        gate_input_dim = hidden_dim * 3  # [F_a_pool; F_e_pool; disagreement]
        self.gate_a = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        self.gate_e = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, F_a, F_e):
        """
        Args:
            F_a: (B, Q, H) — metadata-anchored features
            F_e: (B, Q, H) — exploratory features
        Returns:
            z: (B, H) — fused visual embedding
        """
        # Pool over query dimension
        a_pool = F_a.mean(dim=1)  # (B, H)
        e_pool = F_e.mean(dim=1)  # (B, H)
        # Disagreement signal: proxy for image noise level
        disagreement = (a_pool - e_pool).abs()  # (B, H)

        gate_input = torch.cat([a_pool, e_pool, disagreement], dim=-1)  # (B, 3H)
        g_a = self.gate_a(gate_input)  # (B, 1)
        g_e = self.gate_e(gate_input)  # (B, 1)

        # Normalize gates (soft competition)
        g_sum = g_a + g_e + 1e-8
        g_a = g_a / g_sum
        g_e = g_e / g_sum

        z = g_a * a_pool + g_e * e_pool  # (B, H)
        return self.norm(self.out_proj(z))


class TGQFormer(nn.Module):
    """
    Full TGQ-Former model for multimodal e-commerce I2I retrieval.

    Inputs: (product image visual tokens, product metadata token ids)
    Output: L2-normalized joint embedding for cosine similarity retrieval
    """

    def __init__(
        self,
        visual_dim: int = 2048,       # vision encoder output dim
        hidden_dim: int = 512,
        num_queries: int = 32,
        text_vocab_size: int = 30522,
        text_embed_dim: int = 256,
        embed_dim: int = 256,          # final embedding dimension
    ):
        super().__init__()
        # Project vision encoder output to hidden_dim
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        # Metadata encoder
        self.meta_encoder = MetadataEncoder(text_vocab_size, text_embed_dim, hidden_dim)
        # Hybrid-Query Connector
        self.anchored_query = MetadataAnchoredQuery(num_queries, hidden_dim, num_heads=4)
        self.exploratory_query = ExploratoryQuery(num_queries, hidden_dim, num_heads=4)
        # Reliability-Aware Dual-Gated Modulation
        self.modulation = ReliabilityAwareDualGatedModulation(hidden_dim)
        # Final projection to embedding space
        self.embed_proj = nn.Linear(hidden_dim, embed_dim)
        # Text embedding projection for contrastive alignment
        self.text_embed_proj = nn.Linear(hidden_dim, embed_dim)
        # Temperature for contrastive loss
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def encode_item(self, visual_tokens, meta_ids, meta_mask=None):
        """
        Encode a single product item into a joint embedding.

        Args:
            visual_tokens: (B, N_v, visual_dim) — vision encoder features
            meta_ids:      (B, L)               — metadata token ids
            meta_mask:     (B, L)               — attention mask
        Returns:
            embedding: (B, embed_dim) — L2-normalized
        """
        # Project visual tokens
        V = self.visual_proj(visual_tokens)  # (B, N_v, H)
        # Encode metadata
        M = self.meta_encoder(meta_ids, meta_mask)  # (B, H)
        # Anchored stream: text-guided visual extraction
        F_a = self.anchored_query(V, M)  # (B, Q, H)
        # Exploratory stream: free visual discovery
        F_e = self.exploratory_query(V)  # (B, Q, H)
        # Reliability-aware fusion
        z = self.modulation(F_a, F_e)  # (B, H)
        # Project to embedding space and normalize
        emb = F.normalize(self.embed_proj(z), dim=-1)
        return emb

    def forward(self, visual_tokens_a, meta_ids_a, visual_tokens_b, meta_ids_b,
                meta_mask_a=None, meta_mask_b=None):
        """
        Forward pass for contrastive training (item pairs).
        Returns logits for InfoNCE loss.
        """
        emb_a = self.encode_item(visual_tokens_a, meta_ids_a, meta_mask_a)
        emb_b = self.encode_item(visual_tokens_b, meta_ids_b, meta_mask_b)
        # Cosine similarity matrix (scaled)
        scale = self.logit_scale.exp().clamp(max=100)
        logits = scale * torch.mm(emb_a, emb_b.t())
        return logits, emb_a, emb_b


def infonce_loss(logits):
    """
    InfoNCE / NT-Xent contrastive loss.
    Labels: diagonal elements are positive pairs.
    """
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_i2j = F.cross_entropy(logits, labels)
    loss_j2i = F.cross_entropy(logits.t(), labels)
    return (loss_i2j + loss_j2i) / 2
