"""
QueryAgent-R1: Agentic E-Commerce Query Recommendation with Chain-of-Retrieval RL
Paper: https://arxiv.org/abs/2606.05671
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class QueryAgentConfig:
    vocab_size: int = 32000
    hidden_size: int = 768
    num_layers: int = 6
    num_heads: int = 12
    max_seq_len: int = 512
    memory_size: int = 1024       # number of memory slots
    retrieval_top_k: int = 10     # products to retrieve per query candidate
    num_query_candidates: int = 5  # queries to generate before chain-of-retrieval
    consistency_lambda: float = 0.5  # weight of consistency reward vs. CTR reward


class MemoryModule(nn.Module):
    """Memory bank storing successful query-retrieval-purchase chains."""

    def __init__(self, config: QueryAgentConfig):
        super().__init__()
        self.memory_size = config.memory_size
        self.hidden_size = config.hidden_size
        # Learnable memory keys and values
        self.memory_keys = nn.Parameter(torch.randn(config.memory_size, config.hidden_size))
        self.memory_values = nn.Parameter(torch.randn(config.memory_size, config.hidden_size))
        self.query_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.output_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        """Soft attention over memory given query context."""
        # query: [B, H]
        q = self.query_proj(query)  # [B, H]
        attn = torch.matmul(q, self.memory_keys.T) / (self.hidden_size ** 0.5)  # [B, M]
        attn = F.softmax(attn, dim=-1)
        memory_out = torch.matmul(attn, self.memory_values)  # [B, H]
        return self.output_proj(memory_out)


class UserBehaviorEncoder(nn.Module):
    """Encodes heterogeneous user behavior sequences (views, clicks, searches, purchases)."""

    def __init__(self, config: QueryAgentConfig):
        super().__init__()
        self.item_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        # Behavior type embedding: 0=view, 1=click, 2=search, 3=purchase
        self.behavior_type_emb = nn.Embedding(4, config.hidden_size)
        self.position_emb = nn.Embedding(config.max_seq_len, config.hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size, nhead=config.num_heads,
            dim_feedforward=config.hidden_size * 4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.layer_norm = nn.LayerNorm(config.hidden_size)

    def forward(
        self, item_ids: torch.Tensor, behavior_types: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        item_ids: [B, L] — sequence of item IDs
        behavior_types: [B, L] — 0=view, 1=click, 2=search, 3=purchase
        Returns: [B, H] user representation (CLS token)
        """
        B, L = item_ids.shape
        positions = torch.arange(L, device=item_ids.device).unsqueeze(0).expand(B, -1)
        x = (self.item_embedding(item_ids)
             + self.behavior_type_emb(behavior_types)
             + self.position_emb(positions))
        x = self.layer_norm(x)
        # key_padding_mask expects True where positions should be IGNORED
        key_padding_mask = ~attention_mask.bool() if attention_mask is not None else None
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        return x[:, 0, :]  # [B, H] — CLS representation


class QueryGenerator(nn.Module):
    """Auto-regressive query token generator conditioned on user context + memory."""

    def __init__(self, config: QueryAgentConfig):
        super().__init__()
        self.token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_emb = nn.Embedding(config.max_seq_len, config.hidden_size)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_size, nhead=config.num_heads,
            dim_feedforward=config.hidden_size * 4, batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=config.num_layers)
        self.output_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.context_proj = nn.Linear(config.hidden_size * 2, config.hidden_size)

    def forward(
        self, context: torch.Tensor, tgt_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        context: [B, H] — fused user + memory representation
        tgt_ids: [B, T] — target query token IDs (teacher forcing)
        Returns: [B, T, V] logits
        """
        B, T = tgt_ids.shape
        positions = torch.arange(T, device=tgt_ids.device).unsqueeze(0).expand(B, -1)
        tgt = self.token_emb(tgt_ids) + self.position_emb(positions)
        # Causal mask
        causal_mask = torch.triu(torch.ones(T, T, device=tgt_ids.device), diagonal=1).bool()
        # Memory cross-attention uses context as single-token memory
        memory = context.unsqueeze(1)  # [B, 1, H]
        out = self.transformer(tgt, memory, tgt_mask=causal_mask)
        return self.output_head(out)  # [B, T, V]

    @torch.no_grad()
    def generate(
        self, context: torch.Tensor, max_len: int = 32,
        bos_id: int = 1, eos_id: int = 2
    ) -> torch.Tensor:
        """Greedy decoding for query generation."""
        B = context.size(0)
        device = context.device
        tokens = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        for _ in range(max_len - 1):
            logits = self.forward(context, tokens)  # [B, t, V]
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [B, 1]
            tokens = torch.cat([tokens, next_tok], dim=1)
            if (next_tok == eos_id).all():
                break
        return tokens


class ProductRetriever(nn.Module):
    """Bi-encoder retriever for chain-of-retrieval: encodes query and products."""

    def __init__(self, config: QueryAgentConfig):
        super().__init__()
        self.query_encoder = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        self.product_encoder = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        self.product_emb = nn.Embedding(config.vocab_size, config.hidden_size)

    def encode_query(self, query_repr: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.query_encoder(query_repr), dim=-1)

    def encode_products(self, product_ids: torch.Tensor) -> torch.Tensor:
        """product_ids: [N] — catalog product IDs. Returns [N, H]."""
        return F.normalize(self.product_encoder(self.product_emb(product_ids)), dim=-1)

    def retrieve(
        self, query_repr: torch.Tensor, product_catalog: torch.Tensor, top_k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        query_repr: [B, H]
        product_catalog: [N, H] — all product embeddings
        Returns: (scores [B, top_k], indices [B, top_k])
        """
        q = self.encode_query(query_repr)  # [B, H]
        scores = torch.matmul(q, product_catalog.T)  # [B, N]
        top_scores, top_ids = scores.topk(top_k, dim=-1)
        return top_scores, top_ids


class ConsistencyRewardModule(nn.Module):
    """
    Computes consistency reward: measures alignment between retrieved products
    and the user's actual purchase history.
    Reward = overlap(retrieved_products, purchased_products) / top_k
    """

    def __init__(self, config: QueryAgentConfig):
        super().__init__()
        self.top_k = config.retrieval_top_k

    def forward(
        self, retrieved_ids: torch.Tensor, purchased_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        retrieved_ids: [B, top_k]
        purchased_ids: [B, P] — user's purchase history IDs
        Returns: [B] consistency reward in [0, 1]
        """
        rewards = []
        for b in range(retrieved_ids.size(0)):
            ret_set = set(retrieved_ids[b].tolist())
            pur_set = set(purchased_ids[b].tolist())
            overlap = len(ret_set & pur_set)
            rewards.append(overlap / max(self.top_k, 1))
        return torch.tensor(rewards, dtype=torch.float, device=retrieved_ids.device)


class QueryAgentR1(nn.Module):
    """
    Full QueryAgent-R1 model.

    Architecture:
        UserBehaviorEncoder → fused with MemoryModule → QueryGenerator
        → (generated query) → ProductRetriever → ConsistencyReward
        → RL fine-tuning with GRPO
    """

    def __init__(self, config: QueryAgentConfig):
        super().__init__()
        self.config = config
        self.user_encoder = UserBehaviorEncoder(config)
        self.memory_module = MemoryModule(config)
        self.context_fuse = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.query_generator = QueryGenerator(config)
        self.product_retriever = ProductRetriever(config)
        self.consistency_reward = ConsistencyRewardModule(config)

    def forward(
        self,
        item_ids: torch.Tensor,
        behavior_types: torch.Tensor,
        attention_mask: torch.Tensor,
        tgt_query_ids: torch.Tensor,
        product_catalog_emb: torch.Tensor,
        purchased_ids: torch.Tensor,
    ) -> dict:
        """
        Training forward pass (teacher forcing for SFT; used before RL).
        Returns loss dict.
        """
        # 1. Encode user behavior
        user_repr = self.user_encoder(item_ids, behavior_types, attention_mask)  # [B, H]

        # 2. Retrieve from memory
        mem_repr = self.memory_module(user_repr)  # [B, H]

        # 3. Fuse context
        context = self.context_fuse(torch.cat([user_repr, mem_repr], dim=-1))  # [B, H]

        # 4. Generate query (teacher forcing)
        query_logits = self.query_generator(context, tgt_query_ids[:, :-1])  # [B, T-1, V]
        lm_loss = F.cross_entropy(
            query_logits.reshape(-1, self.config.vocab_size),
            tgt_query_ids[:, 1:].reshape(-1),
            ignore_index=0
        )

        # 5. Chain-of-retrieval: retrieve products for generated query
        with torch.no_grad():
            gen_queries = self.query_generator.generate(context)  # [B, L]
        # Use mean query token embedding as query repr for retrieval
        query_emb = self.query_generator.token_emb(gen_queries).mean(dim=1)  # [B, H]
        _, retrieved_ids = self.product_retriever.retrieve(
            query_emb, product_catalog_emb, self.config.retrieval_top_k
        )

        # 6. Consistency reward
        reward = self.consistency_reward(retrieved_ids, purchased_ids)  # [B]

        return {
            "lm_loss": lm_loss,
            "consistency_reward": reward.mean(),
            "total_loss": lm_loss - self.config.consistency_lambda * reward.mean(),
        }

    def generate_query(
        self,
        item_ids: torch.Tensor,
        behavior_types: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Inference: generate recommended query tokens."""
        user_repr = self.user_encoder(item_ids, behavior_types, attention_mask)
        mem_repr = self.memory_module(user_repr)
        context = self.context_fuse(torch.cat([user_repr, mem_repr], dim=-1))
        return self.query_generator.generate(context)
