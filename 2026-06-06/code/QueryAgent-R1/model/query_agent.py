"""
QueryAgent-R1: Bridging Query Generation and Product Retrieval for E-Commerce Query Recommendation
Paper: https://arxiv.org/abs/2606.05671
Authors: Dike Sun et al., Alibaba International

Architecture:
    User context + Memory → LLM Query Generator → Candidate Queries
    Candidate Queries → Product Retrieval → Retrieved Products
    Retrieved Products ↔ User Preference → Consistency Reward → RL Training

Key formula:
    reward = α · CTR_reward + (1-α) · CVR_consistency_reward
    where CVR_consistency_reward = cosine_sim(mean(product_emb), user_pref_emb)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict


class UserMemoryModule(nn.Module):
    """
    Stores and retrieves user historical queries and interactions.
    Produces a user preference vector for downstream reward computation.
    """

    def __init__(self, item_embed_dim: int = 64, memory_len: int = 20, hidden: int = 128):
        super().__init__()
        self.memory_len = memory_len
        self.item_embed_dim = item_embed_dim

        # Attention-based memory aggregator
        self.query_proj = nn.Linear(item_embed_dim, hidden)
        self.key_proj = nn.Linear(item_embed_dim, hidden)
        self.val_proj = nn.Linear(item_embed_dim, hidden)
        self.out_proj = nn.Linear(hidden, item_embed_dim)

    def forward(self, memory: torch.Tensor, current_context: torch.Tensor) -> torch.Tensor:
        """
        memory:           (B, M, D)   past interacted item embeddings
        current_context:  (B, D)      current session context

        Returns user preference vector: (B, D)
        """
        q = self.query_proj(current_context.unsqueeze(1))  # (B, 1, H)
        k = self.key_proj(memory)                          # (B, M, H)
        v = self.val_proj(memory)                          # (B, M, H)

        attn = (q @ k.transpose(-2, -1)) / (k.shape[-1] ** 0.5)  # (B, 1, M)
        attn = F.softmax(attn, dim=-1)

        aggregated = (attn @ v).squeeze(1)  # (B, H)
        return self.out_proj(aggregated)    # (B, D)


class QueryGenerator(nn.Module):
    """
    LLM-based query generator.
    In the full system, this wraps a frozen/LoRA-tuned LLM (e.g., Qwen2.5).
    Here, we implement a lightweight Transformer-based proxy.

    Input: user preference vector + context token sequence
    Output: query token logits (greedy decode → query string)
    """

    def __init__(
        self,
        vocab_size: int = 10_000,
        embed_dim: int = 64,
        hidden: int = 256,
        num_layers: int = 2,
        max_query_len: int = 16,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_query_len = max_query_len

        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.context_proj = nn.Linear(embed_dim, hidden)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=4, dim_feedforward=512, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden, nhead=4, dim_feedforward=512, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.out = nn.Linear(hidden, vocab_size)

    def forward(
        self,
        user_pref: torch.Tensor,     # (B, embed_dim) user preference vector
        tgt_ids: Optional[torch.Tensor] = None,  # (B, L) for teacher forcing
    ) -> torch.Tensor:
        B = user_pref.shape[0]
        # Context = user preference vector projected to hidden
        memory = self.context_proj(user_pref).unsqueeze(1)  # (B, 1, H)
        memory = self.encoder(memory)

        if tgt_ids is not None:
            tgt_emb = self.context_proj(self.token_embed(tgt_ids))  # (B, L, H)
        else:
            # Greedy decode: start with a [BOS] token (id=1)
            tgt_ids = torch.ones(B, 1, dtype=torch.long, device=user_pref.device)
            tgt_emb = self.context_proj(self.token_embed(tgt_ids))  # (B, 1, H)

        out = self.decoder(tgt_emb, memory)
        logits = self.out(out)  # (B, L, vocab_size)
        return logits

    def generate(self, user_pref: torch.Tensor, max_len: int = 8) -> torch.Tensor:
        """Greedy query generation. Returns (B, max_len) token IDs."""
        B = user_pref.shape[0]
        device = user_pref.device
        memory = self.context_proj(user_pref).unsqueeze(1)
        memory = self.encoder(memory)

        generated = torch.ones(B, 1, dtype=torch.long, device=device)  # [BOS]
        for _ in range(max_len - 1):
            tgt_emb = self.context_proj(self.token_embed(generated))
            out = self.decoder(tgt_emb, memory)
            logits = self.out(out[:, -1, :])  # (B, vocab)
            next_token = logits.argmax(dim=-1, keepdim=True)  # (B, 1)
            generated = torch.cat([generated, next_token], dim=1)
        return generated  # (B, max_len)


class QueryAgentR1(nn.Module):
    """
    Full QueryAgent-R1 model.

    Components:
        - UserMemoryModule: builds user preference vector from interaction history
        - QueryGenerator: generates candidate query tokens
    """

    def __init__(
        self,
        item_embed_dim: int = 64,
        vocab_size: int = 10_000,
        memory_len: int = 20,
    ):
        super().__init__()
        self.memory_module = UserMemoryModule(item_embed_dim=item_embed_dim, memory_len=memory_len)
        self.query_generator = QueryGenerator(
            vocab_size=vocab_size, embed_dim=item_embed_dim
        )

    def forward(
        self,
        interaction_history: torch.Tensor,  # (B, M, D)
        current_context: torch.Tensor,      # (B, D)
        tgt_ids: Optional[torch.Tensor] = None,  # for training
    ) -> Dict[str, torch.Tensor]:
        user_pref = self.memory_module(interaction_history, current_context)  # (B, D)
        query_logits = self.query_generator(user_pref, tgt_ids)  # (B, L, vocab)
        return {
            "user_pref": user_pref,
            "query_logits": query_logits,
        }

    def generate_queries(
        self,
        interaction_history: torch.Tensor,
        current_context: torch.Tensor,
        max_len: int = 8,
    ) -> torch.Tensor:
        user_pref = self.memory_module(interaction_history, current_context)
        return self.query_generator.generate(user_pref, max_len)
