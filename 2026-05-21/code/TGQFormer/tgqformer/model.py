"""TGQ-Former: Text-Guided Q-Former for robust multimodal e-commerce item retrieval.

Core components (faithful to arXiv 2605.17366):
  1. ImageEncoder    — frozen patch encoder (lightweight CNN for toy)
  2. TextEncoder     — embeds structured metadata tokens
  3. HybridQueryQFormer — two query streams:
       * MetadataStream : text-guided cross-attn over image patches
       * ExploratoryStream: free self-attn over image patches (no text constraint)
  4. RDGVM (Reliability-Aware Dual-Gated Vector Modulation) — adaptively gates
       the two streams based on estimated visual reliability
  5. Projection head  — projects fused embedding to retrieval space
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Patch encoder (toy CNN replaces ViT)
# ---------------------------------------------------------------------------

class ImageEncoder(nn.Module):
    """Lightweight CNN that outputs patch-level feature tokens."""

    def __init__(self, in_ch: int = 3, d_model: int = 64, num_patches: int = 16):
        super().__init__()
        # Map (B, 3, H, W) -> (B, num_patches, d_model)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, d_model, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(d_model, d_model, 3, stride=2, padding=1), nn.GELU(),
        )
        self.num_patches = num_patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) -> patches: (B, num_patches, d_model)"""
        h = self.conv(x)                          # (B, d, H', W')
        B, d, H, W = h.shape
        h = h.flatten(2).transpose(1, 2)          # (B, H'*W', d)
        if h.size(1) != self.num_patches:
            h = F.adaptive_avg_pool1d(h.transpose(1, 2), self.num_patches).transpose(1, 2)
        return h


# ---------------------------------------------------------------------------
# Text encoder
# ---------------------------------------------------------------------------

class TextEncoder(nn.Module):
    """Simple embedding + transformer for structured metadata."""

    def __init__(self, vocab_size: int = 512, d_model: int = 64, seq_len: int = 16):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = nn.Embedding(seq_len, d_model)
        self.tf = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=4, dim_feedforward=128,
                                       batch_first=True, norm_first=True),
            num_layers=2,
        )
        self.seq_len = seq_len

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, L) -> text_emb: (B, L, d_model)"""
        B, L = tokens.shape
        pos = torch.arange(L, device=tokens.device).unsqueeze(0).expand(B, -1)
        x = self.emb(tokens) + self.pos(pos)
        return self.tf(x)


# ---------------------------------------------------------------------------
# Cross-attention block (shared utility)
# ---------------------------------------------------------------------------

class CrossAttention(nn.Module):
    def __init__(self, d_model: int = 64, nhead: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """q: (B, Nq, d), kv: (B, Nkv, d) -> (B, Nq, d)"""
        out, _ = self.attn(self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv))
        return q + out


# ---------------------------------------------------------------------------
# Hybrid-Query Q-Former
# ---------------------------------------------------------------------------

class HybridQueryQFormer(nn.Module):
    """Two parallel query streams over image patches.

    MetadataStream: query initialized from text embedding (metadata-anchored).
    ExploratoryStream: query initialized from learnable tokens (unconstrained).

    Both streams perform cross-attention over image patch tokens.

    Eq (paper §3.2, simplified):
        h_meta   = CrossAttn(Q_meta, patches)     Q_meta ← text_emb pooled
        h_explore = CrossAttn(Q_explore, patches)  Q_explore ← learned tokens
    """

    def __init__(self, d_model: int = 64, num_query: int = 8, nhead: int = 4):
        super().__init__()
        self.num_query = num_query
        # Metadata stream
        self.meta_proj = nn.Linear(d_model, d_model)  # project text -> meta queries
        self.meta_xattn = CrossAttention(d_model, nhead)
        self.meta_self_attn = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=128, batch_first=True, norm_first=True
        )
        # Exploratory stream
        self.explore_queries = nn.Parameter(torch.randn(1, num_query, d_model) * 0.02)
        self.explore_xattn = CrossAttention(d_model, nhead)
        self.explore_self_attn = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=128, batch_first=True, norm_first=True
        )

    def forward(
        self, patches: torch.Tensor, text_emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        patches  : (B, Np, d)
        text_emb : (B, L, d)
        Returns:
            h_meta     : (B, num_query, d)
            h_explore  : (B, num_query, d)
        """
        B = patches.size(0)

        # --- Metadata stream: init queries from mean-pooled text ---
        text_pool = text_emb.mean(dim=1, keepdim=True)  # (B, 1, d)
        meta_q = self.meta_proj(text_pool).expand(-1, self.num_query, -1)  # (B, Nq, d)
        h_meta = self.meta_xattn(meta_q, patches)
        h_meta = self.meta_self_attn(h_meta)

        # --- Exploratory stream: init queries from learnable tokens ---
        explore_q = self.explore_queries.expand(B, -1, -1)
        h_explore = self.explore_xattn(explore_q, patches)
        h_explore = self.explore_self_attn(h_explore)

        return h_meta, h_explore


# ---------------------------------------------------------------------------
# RDGVM: Reliability-Aware Dual-Gated Vector Modulation
# ---------------------------------------------------------------------------

class RDGVM(nn.Module):
    """Reliability-Aware Dual-Gated Vector Modulation.

    Estimates visual reliability from patch statistics and gates the
    contributions of the metadata-anchored and exploratory streams.

    When the image is noisy (large promotional overlay), the gate should
    up-weight the metadata stream and down-weight the exploratory stream.

    Gate computation (Eq. paper §3.3, simplified):
        r  = sigmoid(W_r · [var(patches), entropy(patches)])
        g_meta    = sigmoid(W_m · r)
        g_explore = sigmoid(W_e · (1 - r))
        h_fused   = g_meta * h_meta + g_explore * h_explore
    """

    def __init__(self, d_model: int = 64):
        super().__init__()
        # Reliability estimator: takes patch-level statistics
        self.reliability_net = nn.Sequential(
            nn.Linear(2, 32), nn.GELU(), nn.Linear(32, d_model), nn.Sigmoid()
        )
        # Per-stream gates
        self.gate_meta = nn.Linear(d_model, d_model)
        self.gate_explore = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def _patch_reliability(self, patches: torch.Tensor) -> torch.Tensor:
        """Compute per-sample reliability score from patch variance and entropy.

        patches: (B, Np, d) -> reliability: (B, d)
        """
        # Spatial variance (low variance = uniform overlay = less reliable)
        var = patches.var(dim=1)                     # (B, d)
        var_mean = var.mean(dim=-1, keepdim=True)     # (B, 1)

        # Entropy of mean-pooled softmax distribution (proxy for patch diversity)
        p = F.softmax(patches.mean(dim=-1), dim=-1)  # (B, Np)
        entropy = -(p * (p + 1e-8).log()).sum(dim=-1, keepdim=True)  # (B, 1)
        entropy = entropy / math.log(patches.size(1) + 1e-8)         # normalize

        feat = torch.cat([var_mean, entropy], dim=-1)  # (B, 2)
        return self.reliability_net(feat)               # (B, d)

    def forward(
        self,
        h_meta: torch.Tensor,
        h_explore: torch.Tensor,
        patches: torch.Tensor,
    ) -> torch.Tensor:
        """
        h_meta    : (B, Nq, d)
        h_explore : (B, Nq, d)
        patches   : (B, Np, d)
        Returns fused embedding: (B, Nq, d)
        """
        r = self._patch_reliability(patches)      # (B, d)
        r = r.unsqueeze(1)                        # (B, 1, d)

        g_meta = torch.sigmoid(self.gate_meta(r))      # (B, 1, d)
        g_explore = torch.sigmoid(self.gate_explore(1 - r))

        h_fused = g_meta * h_meta + g_explore * h_explore
        return self.norm(h_fused)


# ---------------------------------------------------------------------------
# Full TGQ-Former model
# ---------------------------------------------------------------------------

class TGQFormer(nn.Module):
    """TGQ-Former for e-commerce I2I retrieval.

    Forward pass returns a normalized embedding vector per item.
    Training uses contrastive loss over (query, positive, negative) triplets.
    """

    def __init__(
        self,
        vocab_size: int = 512,
        d_model: int = 64,
        num_patches: int = 16,
        num_query: int = 8,
        text_seq_len: int = 16,
        emb_dim: int = 128,
        freeze_vision: bool = False,
    ):
        super().__init__()
        self.img_enc = ImageEncoder(d_model=d_model, num_patches=num_patches)
        self.txt_enc = TextEncoder(vocab_size=vocab_size, d_model=d_model,
                                   seq_len=text_seq_len)
        self.hybrid_qformer = HybridQueryQFormer(d_model=d_model, num_query=num_query)
        self.rdgvm = RDGVM(d_model=d_model)
        # Project fused tokens to retrieval embedding
        self.proj = nn.Sequential(
            nn.Linear(d_model * num_query, 256), nn.GELU(),
            nn.Linear(256, emb_dim),
        )
        if freeze_vision:
            for p in self.img_enc.parameters():
                p.requires_grad = False

    def encode_item(
        self, image: torch.Tensor, text_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Encode a single item to a retrieval embedding.

        image       : (B, C, H, W)
        text_tokens : (B, L)
        Returns     : (B, emb_dim) L2-normalized
        """
        patches = self.img_enc(image)           # (B, Np, d)
        text_emb = self.txt_enc(text_tokens)    # (B, L, d)
        h_meta, h_explore = self.hybrid_qformer(patches, text_emb)
        h_fused = self.rdgvm(h_meta, h_explore, patches)  # (B, Nq, d)
        emb = self.proj(h_fused.flatten(1))     # (B, emb_dim)
        return F.normalize(emb, dim=-1)

    def forward(
        self,
        q_image: torch.Tensor, q_text: torch.Tensor,
        pos_image: torch.Tensor, pos_text: torch.Tensor,
        neg_image: torch.Tensor, neg_text: torch.Tensor,
    ) -> dict:
        """Return embeddings for query, positive, and negative items."""
        q_emb = self.encode_item(q_image, q_text)
        pos_emb = self.encode_item(pos_image, pos_text)
        neg_emb = self.encode_item(neg_image, neg_text)
        return {"q": q_emb, "pos": pos_emb, "neg": neg_emb}
