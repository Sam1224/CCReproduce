"""
QueryAgent-R1 core model components.
Faithful reproduction of arXiv:2606.05671.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple


class InterestNode(nn.Module):
    """Single interest node in the Interest Graph."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, item_embeds: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # Weighted aggregation of item embeddings within a cluster
        return (item_embeds * weights.unsqueeze(-1)).sum(dim=0)


class InterestGraphMemory(nn.Module):
    """
    Memory abstraction module that extracts an interest graph from a user's
    long-term interaction history for efficient user profiling (§3.2 of paper).

    The interest graph clusters historical items into K interest nodes,
    compressing an arbitrarily long history into a fixed-size representation.
    """

    def __init__(
        self,
        item_embed_dim: int,
        num_interest_nodes: int = 8,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.item_embed_dim = item_embed_dim
        self.num_interest_nodes = num_interest_nodes
        self.hidden_dim = hidden_dim

        # Item encoder (projects raw item embedding to shared space)
        self.item_encoder = nn.Sequential(
            nn.Linear(item_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Soft assignment: each item is assigned to K interest nodes
        self.assignment_head = nn.Linear(hidden_dim, num_interest_nodes)

        # Interest node refinement
        self.node_refiner = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
        )

        # Final user embedding projection
        self.user_projector = nn.Sequential(
            nn.Linear(hidden_dim * num_interest_nodes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(
        self, item_embeds: torch.Tensor, item_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            item_embeds: (B, L, D) — batch of user histories, L items each
            item_mask: (B, L) — 1 for valid items, 0 for padding

        Returns:
            user_embed: (B, hidden_dim) — compressed user representation
            interest_nodes: (B, K, hidden_dim) — K interest node embeddings
        """
        B, L, _ = item_embeds.shape

        # Encode items
        h = self.item_encoder(item_embeds)  # (B, L, hidden_dim)

        # Compute soft assignments to K interest nodes
        logits = self.assignment_head(h)  # (B, L, K)
        if item_mask is not None:
            logits = logits.masked_fill(~item_mask.unsqueeze(-1).bool(), float("-inf"))
        assignments = F.softmax(logits, dim=1)  # (B, L, K) — node weights per item

        # Aggregate: K interest nodes as weighted sums over L items
        # interest_nodes: (B, K, hidden_dim)
        interest_nodes = torch.bmm(assignments.transpose(1, 2), h)

        # Refine interest nodes with self-attention
        interest_nodes = self.node_refiner(interest_nodes)  # (B, K, hidden_dim)

        # Flatten and project to user embedding
        user_embed = self.user_projector(
            interest_nodes.reshape(B, self.num_interest_nodes * self.hidden_dim)
        )  # (B, hidden_dim)

        return user_embed, interest_nodes


class QueryGenerator(nn.Module):
    """
    LLM-based query generator conditioned on user memory and retrieved context.
    Uses a pre-trained language model backbone for query generation.

    In practice (paper), this is a fine-tuned large language model.
    Here we implement a lightweight Transformer-based surrogate.
    """

    def __init__(
        self,
        vocab_size: int,
        user_embed_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        max_query_len: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_query_len = max_query_len

        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embed = nn.Embedding(max_query_len + 2, hidden_dim)

        # User context injection via cross-attention
        self.user_projector = nn.Linear(user_embed_dim, hidden_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_head = nn.Linear(hidden_dim, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        input_ids: torch.Tensor,
        user_embed: torch.Tensor,
        retrieved_context: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (B, T) — partial query token IDs
            user_embed: (B, user_embed_dim) — user interest embedding
            retrieved_context: (B, R, hidden_dim) — embeddings of retrieved products (optional)
            attention_mask: (B, T) causal mask

        Returns:
            logits: (B, T, vocab_size)
        """
        B, T = input_ids.shape

        # Token embeddings
        pos_ids = torch.arange(T, device=input_ids.device).unsqueeze(0)
        tok_emb = self.token_embed(input_ids) + self.pos_embed(pos_ids)

        # Memory: project user embedding as cross-attention target
        user_mem = self.user_projector(user_embed).unsqueeze(1)  # (B, 1, hidden_dim)

        if retrieved_context is not None:
            memory = torch.cat([user_mem, retrieved_context], dim=1)
        else:
            memory = user_mem

        # Causal mask for autoregressive generation
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=input_ids.device)

        out = self.decoder(
            tgt=tok_emb,
            memory=memory,
            tgt_mask=causal_mask,
        )  # (B, T, hidden_dim)

        logits = self.output_head(out)  # (B, T, vocab_size)
        return logits

    @torch.no_grad()
    def generate(
        self,
        user_embed: torch.Tensor,
        bos_token_id: int,
        eos_token_id: int,
        retrieved_context: Optional[torch.Tensor] = None,
        max_new_tokens: int = 16,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Greedy / temperature-sampled generation."""
        B = user_embed.shape[0]
        generated = torch.full((B, 1), bos_token_id, dtype=torch.long, device=user_embed.device)

        for _ in range(max_new_tokens):
            logits = self.forward(generated, user_embed, retrieved_context)
            next_logits = logits[:, -1, :] / temperature
            next_token = next_logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

            if (next_token == eos_token_id).all():
                break

        return generated


class QueryAgentR1(nn.Module):
    """
    Full QueryAgent-R1 model.

    Combines:
    - InterestGraphMemory for efficient user profiling
    - QueryGenerator for query generation conditioned on user memory + retrieved products
    - ConsistencyReward (in reward.py) for RL training signal
    """

    def __init__(
        self,
        item_embed_dim: int,
        vocab_size: int,
        num_interest_nodes: int = 8,
        hidden_dim: int = 256,
        max_query_len: int = 32,
        num_gen_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.memory = InterestGraphMemory(
            item_embed_dim=item_embed_dim,
            num_interest_nodes=num_interest_nodes,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.generator = QueryGenerator(
            vocab_size=vocab_size,
            user_embed_dim=hidden_dim,
            hidden_dim=hidden_dim,
            max_query_len=max_query_len,
            num_layers=num_gen_layers,
            dropout=dropout,
        )

    def encode_user(
        self, item_embeds: torch.Tensor, item_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.memory(item_embeds, item_mask)

    def generate_query(
        self,
        user_embed: torch.Tensor,
        bos_token_id: int,
        eos_token_id: int,
        retrieved_context: Optional[torch.Tensor] = None,
        max_new_tokens: int = 16,
    ) -> torch.Tensor:
        return self.generator.generate(
            user_embed, bos_token_id, eos_token_id, retrieved_context, max_new_tokens
        )

    def forward(
        self,
        item_embeds: torch.Tensor,
        query_input_ids: torch.Tensor,
        item_mask: Optional[torch.Tensor] = None,
        retrieved_context: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Teacher-forcing forward for training."""
        user_embed, interest_nodes = self.memory(item_embeds, item_mask)
        logits = self.generator(query_input_ids, user_embed, retrieved_context)
        return {
            "logits": logits,
            "user_embed": user_embed,
            "interest_nodes": interest_nodes,
        }
