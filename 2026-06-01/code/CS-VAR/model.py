"""
CS-VAR: Cross-Session Evidence-Aware Retrieval-Augmented Detector
Reproduction of: "Deja Vu in Plots: Leveraging Cross-Session Evidence with
Retrieval-Augmented LLMs for Live Streaming Risk Assessment"
arXiv: 2601.16027

Core idea (paper §3):
  Training phase:
    1. For each session, retrieve k similar historical sessions from database.
    2. LLM teacher reasons over (current session + retrieved history) →
       structured risk assessment with local-to-global pattern recognition.
    3. Distill LLM teacher knowledge into a lightweight SessionRiskModel.
  Inference phase:
    - Only the lightweight SessionRiskModel runs; no LLM call needed.
    - O(1) latency for real-time deployment.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SessionEncoder(nn.Module):
    """
    Encodes a livestream session into a fixed-size representation.

    A session consists of temporal behavioral events (e.g., comments, gifts,
    host actions, product interactions). We model this as a sequence.

    Paper models sessions as sequences of behavioral events; we approximate
    with a Transformer encoder over event features.
    """

    def __init__(
        self,
        event_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        max_seq_len: int = 128,
    ):
        super().__init__()
        self.embed = nn.Linear(event_dim, hidden_dim)
        self.pos_embed = nn.Embedding(max_seq_len, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, events, mask=None):
        """
        Args:
            events: (B, T, event_dim) — sequence of behavioral events
            mask:   (B, T) — True for padding positions
        Returns:
            session_repr: (B, hidden_dim)
        """
        B, T, _ = events.shape
        pos = torch.arange(T, device=events.device).unsqueeze(0)
        x = self.embed(events) + self.pos_embed(pos)
        x = self.transformer(x, src_key_padding_mask=mask)
        # CLS-style pooling: mean over non-masked positions
        if mask is not None:
            valid = (~mask).float().unsqueeze(-1)  # (B, T, 1)
            x = (x * valid).sum(1) / valid.sum(1).clamp(min=1)
        else:
            x = x.mean(dim=1)
        return self.norm(self.pool(x))


class CrossSessionFusionModule(nn.Module):
    """
    Fuses current session representation with retrieved cross-session evidence.

    Paper §3.2: "the LLM transfers local-to-global insights to the small model"
    → We approximate this with a cross-attention module.

    At training time: enriched with LLM reasoning output (via KD).
    At inference time: runs purely on retrieved session embeddings.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, query, key_value):
        """
        Args:
            query:     (B, 1, H) — current session
            key_value: (B, K, H) — K retrieved historical sessions
        Returns:
            fused: (B, H)
        """
        attn_out, _ = self.cross_attn(query, key_value, key_value)
        x = self.norm(query + attn_out)
        x = self.norm(x + self.ffn(x))
        return x.squeeze(1)  # (B, H)


class SessionRiskModel(nn.Module):
    """
    Lightweight student model for real-time live streaming risk assessment.

    Components:
      1. SessionEncoder: encodes event sequence → session embedding
      2. CrossSessionFusionModule: integrates retrieved cross-session evidence
      3. Risk head: predicts risk score / category
    """

    def __init__(
        self,
        event_dim: int = 64,
        hidden_dim: int = 256,
        num_risk_levels: int = 3,  # e.g., Low / Medium / High
    ):
        super().__init__()
        self.encoder = SessionEncoder(event_dim, hidden_dim)
        self.cs_fusion = CrossSessionFusionModule(hidden_dim)
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_risk_levels),
        )
        self.hidden_dim = hidden_dim

    def encode_session(self, events, mask=None):
        return self.encoder(events, mask)

    def forward(self, current_events, retrieved_embeddings, mask=None):
        """
        Args:
            current_events:        (B, T, event_dim)
            retrieved_embeddings:  (B, K, hidden_dim) — pre-encoded retrieved sessions
            mask:                  (B, T) padding mask
        Returns:
            logits: (B, num_risk_levels)
            emb:    (B, hidden_dim) — session embedding for retrieval database
        """
        # Encode current session
        curr_emb = self.encoder(current_events, mask)  # (B, H)

        # Fuse with cross-session evidence
        curr_query = curr_emb.unsqueeze(1)  # (B, 1, H)
        fused = self.cs_fusion(curr_query, retrieved_embeddings)  # (B, H)

        # Concatenate original and fused for final prediction
        combined = torch.cat([curr_emb, fused], dim=-1)  # (B, 2H)
        logits = self.risk_head(combined)

        return logits, curr_emb
