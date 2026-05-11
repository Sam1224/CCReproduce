"""
GLiGuard — Model Implementation
Schema-conditioned bidirectional encoder for multi-aspect LLM safety classification.

Paper: "GLiGuard: Schema-Conditioned Classification for LLM Safeguard"
       (arXiv 2605.07982)

Architecture:
  - Backbone: 0.3B bidirectional encoder (GLiNER2 style; toy: small BERT-like)
  - Schema conditioning: safety dimension names are embedded and fused with input
  - Multi-task head: parallel binary classification for each active schema dimension
  - Single forward pass evaluates all active dimensions simultaneously

Key distinction from autoregressive guardrails (LlamaGuard):
  - Encoder is bidirectional → captures late-appearing harm signals
  - Classification (not generation) → 16-17× throughput/latency improvement
  - Schema composition → flexible dimension selection without prompt re-engineering
"""

from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import ALL_SCHEMA_DIMS


# ---------------------------------------------------------------------------
# Schema Embedding Module
# Paper: schema dimensions are embedded and appended to input for conditioning
# ---------------------------------------------------------------------------

class SchemaEmbedder(nn.Module):
    """
    Embeds safety schema dimension names as learnable vectors.

    In the paper, GLiNER2 uses span-level entity type conditioning.
    Here, we adapt this for safety schema dimensions.
    """

    def __init__(self, schema_dims: List[str], hidden_dim: int = 256):
        super().__init__()
        self.schema_dims = schema_dims
        self.dim2idx = {d: i for i, d in enumerate(schema_dims)}
        # Learnable embedding for each schema dimension
        self.schema_emb = nn.Embedding(len(schema_dims), hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, active_schemas: List[List[str]]) -> List[torch.Tensor]:
        """
        Returns schema embeddings for each sample in the batch.
        Each sample gets a list of embeddings (one per active schema dim).
        """
        result = []
        for schema_list in active_schemas:
            indices = torch.tensor(
                [self.dim2idx.get(d, 0) for d in schema_list],
                dtype=torch.long,
                device=self.schema_emb.weight.device,
            )
            emb = self.schema_emb(indices)        # (num_active, hidden)
            emb = self.proj(emb)
            result.append(emb)
        return result


# ---------------------------------------------------------------------------
# Bidirectional Encoder (GLiNER2-style backbone)
# Paper: "a 0.3B-parameter schema-conditioned bidirectional encoder adapted
#         from GLiNER2"
# Toy: lightweight 2-layer Transformer encoder
# ---------------------------------------------------------------------------

class BidirectionalEncoder(nn.Module):
    """
    Lightweight bidirectional encoder backbone.

    Production: load GLiNER2 (deberta-v3-base or similar) pretrained weights.
    Toy: 2-layer Transformer encoder with sinusoidal positional encoding.
    """

    def __init__(
        self,
        vocab_size: int = 8192,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Token embedding
        self.token_emb = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)
        self.emb_dropout = nn.Dropout(dropout)

        # Transformer encoder (bidirectional by default in PyTorch)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,          # Pre-LN for stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def tokenize(self, texts: List[str], max_len: int = 256) -> torch.Tensor:
        """Toy tokenizer: character-level hashing."""
        batch = []
        for text in texts:
            ids = [ord(c) % 8191 + 1 for c in text[:max_len]]
            ids += [0] * (max_len - len(ids))
            batch.append(ids)
        return torch.tensor(batch, dtype=torch.long)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (B, L) token IDs
        Returns:
            hidden: (B, L, hidden_dim) contextual representations
        """
        B, L = input_ids.shape
        device = input_ids.device

        tok_emb = self.token_emb(input_ids)
        pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_emb(pos_ids)

        x = self.emb_dropout(tok_emb + pos_emb)
        pad_mask = (input_ids == 0)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        return self.norm(x)


# ---------------------------------------------------------------------------
# GLiGuard: Main Model
# ---------------------------------------------------------------------------

class GLiGuard(nn.Module):
    """
    Full GLiGuard model: schema-conditioned multi-aspect safety classifier.

    Paper architecture:
      1. Concatenate prompt + response as input sequence
      2. Insert schema dimension tokens (as span markers from GLiNER2)
      3. Bidirectional encoder processes the full sequence
      4. Per-schema classification head reads the schema-conditioned representation
      5. All active dimensions classified in ONE forward pass

    Complexity: O(L²) per forward pass (standard attention), NOT O(L × D) as in
    autoregressive generation where D = num output tokens per dimension.
    """

    def __init__(
        self,
        schema_dims: Optional[List[str]] = None,
        hidden_dim: int = 256,
        num_encoder_layers: int = 2,
        dropout: float = 0.1,
        device: str = "cpu",
    ):
        super().__init__()
        self.schema_dims = schema_dims or ALL_SCHEMA_DIMS
        self.device = device

        self.encoder = BidirectionalEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_encoder_layers,
            dropout=dropout,
        )

        self.schema_embedder = SchemaEmbedder(
            schema_dims=self.schema_dims,
            hidden_dim=hidden_dim,
        )

        # Cross-attention: schema dim embeddings attend to encoder output
        self.schema_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )

        # Per-dimension binary classification head (shared MLP)
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),   # binary: safe / unsafe
        )

    def forward(
        self,
        prompts: List[str],
        responses: List[str],
        active_schemas: List[List[str]],
    ) -> Dict[str, torch.Tensor]:
        """
        Single forward pass evaluating all active schema dimensions.

        Args:
            prompts: (B,) prompt texts
            responses: (B,) response texts
            active_schemas: (B,) list of active schema dimension names per sample

        Returns:
            dict mapping schema_dim → logits (B_active, 2)
            (only samples where the dim is active contribute)
        """
        device = next(self.parameters()).device

        # Concatenate prompt + response for encoder input
        combined = [f"[PROMPT] {p} [RESPONSE] {r}" for p, r in zip(prompts, responses)]
        input_ids = self.encoder.tokenize(combined).to(device)

        # Encode full sequence (bidirectional)
        hidden = self.encoder(input_ids)      # (B, L, D)

        # CLS representation: mean pooling
        cls_rep = hidden.mean(dim=1)          # (B, D)

        # Schema embedding for each sample
        schema_embs = self.schema_embedder(active_schemas)  # List of (num_active_i, D)

        # Per-sample, per-active-dim cross-attention + classification
        dim_logits: Dict[str, List[torch.Tensor]] = {}

        for b_idx, (sample_embs, sample_schema) in enumerate(zip(schema_embs, active_schemas)):
            if len(sample_schema) == 0:
                continue

            # Cross-attention: schema dims attend to encoder hidden states
            # Query: schema embeddings, Key/Value: encoder hidden
            h = hidden[b_idx].unsqueeze(0)        # (1, L, D)
            q = sample_embs.unsqueeze(0)           # (1, num_active, D)
            attn_out, _ = self.schema_cross_attn(q, h, h)  # (1, num_active, D)
            attn_out = attn_out.squeeze(0)         # (num_active, D)

            # Residual connection with CLS rep
            cls = cls_rep[b_idx].unsqueeze(0).expand_as(attn_out)
            fused = attn_out + cls                 # (num_active, D)

            # Classification for each active dim
            logits = self.cls_head(fused)          # (num_active, 2)

            for dim_idx, dim_name in enumerate(sample_schema):
                if dim_name not in dim_logits:
                    dim_logits[dim_name] = []
                dim_logits[dim_name].append(logits[dim_idx])

        # Stack per-dim logits
        return {
            dim: torch.stack(logit_list, dim=0)
            for dim, logit_list in dim_logits.items()
        }

    def predict(
        self,
        prompts: List[str],
        responses: List[str],
        active_schemas: List[List[str]],
        threshold: float = 0.5,
    ) -> List[Dict[str, int]]:
        """
        Returns per-sample, per-dim binary predictions.
        """
        self.eval()
        with torch.no_grad():
            dim_logits = self.forward(prompts, responses, active_schemas)

        B = len(prompts)
        predictions = [{} for _ in range(B)]

        # Reconstruct per-sample predictions from per-dim logit lists
        dim_sample_idx: Dict[str, int] = {}
        for b_idx, schema in enumerate(active_schemas):
            for dim in schema:
                if dim not in dim_sample_idx:
                    dim_sample_idx[dim] = 0
                logits = dim_logits.get(dim)
                if logits is None:
                    continue
                idx = dim_sample_idx[dim]
                if idx < len(logits):
                    prob = F.softmax(logits[idx], dim=-1)[1].item()
                    predictions[b_idx][dim] = int(prob > threshold)
                    dim_sample_idx[dim] += 1

        return predictions
