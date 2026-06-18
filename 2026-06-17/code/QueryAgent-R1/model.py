"""
QueryAgent-R1 — Model Architecture

Key components (faithful to the paper):
  1. QueryEncoder       — encodes query sequence into user preference embedding
  2. Retriever          — simulated retrieval engine (returns product embeddings)
  3. ChainOfRetrievalOptimizer — Agent: generate query → retrieve → score → optimize
  4. R1-style reward    — CVR signal as reward for RL training

Paper formula references:
  Query generation:  q* = argmax_q CVR(Retriever(q), user)     (Eq. 1)
  Agent reward:      R = CVR_signal(retrieved_products, user)   (Eq. 2)
  Policy gradient:   L_RL = -R * log P(q* | seq)               (Eq. 3 — REINFORCE)
  CTR alignment:     L_CTR = -log P(next_query | seq)           (Eq. 4)
  Total:             L = L_CTR + λ * L_RL                       (Eq. 5)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class QueryEncoder(nn.Module):
    """
    Encodes user query history into a preference embedding.
    Architecture: Embedding + Transformer (SASRec-style)
    """

    def __init__(self, n_queries: int, embed_dim: int = 64, hidden_dim: int = 128,
                 n_heads: int = 4, n_layers: int = 2, max_seq_len: int = 10):
        super().__init__()
        self.query_emb = nn.Embedding(n_queries + 1, embed_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=hidden_dim,
            batch_first=True, dropout=0.1, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.out_proj = nn.Linear(embed_dim, hidden_dim)

    def forward(self, query_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query_seq: (B, L) query IDs
        Returns:
            user_pref: (B, hidden_dim)
        """
        B, L = query_seq.shape
        pos = torch.arange(L, device=query_seq.device).unsqueeze(0)
        x = self.query_emb(query_seq) + self.pos_emb(pos)
        pad_mask = (query_seq == 0)
        x = self.transformer(x, src_key_padding_mask=pad_mask)
        mask = (~pad_mask).float().unsqueeze(-1)
        x = (x * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.out_proj(x)


class QueryProjectionHead(nn.Module):
    """
    Projects user preference into query space for generation scoring.
    In the paper, a generative LLM head is used; here we approximate with
    a linear projection over a fixed query vocabulary.
    """

    def __init__(self, hidden_dim: int, n_queries: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, n_queries)

    def forward(self, user_pref: torch.Tensor) -> torch.Tensor:
        """Returns logits over query vocabulary: (B, n_queries)"""
        return self.out(F.gelu(self.proj(user_pref)))


class MemoryModule(nn.Module):
    """
    Memory augmentation for long-term user preference.
    Stores a fixed-size memory bank of past query embeddings.
    """

    def __init__(self, hidden_dim: int, memory_size: int = 16):
        super().__init__()
        self.memory_size = memory_size
        self.memory_key = nn.Parameter(torch.randn(memory_size, hidden_dim))
        self.memory_val = nn.Parameter(torch.randn(memory_size, hidden_dim))
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)

    def forward(self, user_pref: torch.Tensor) -> torch.Tensor:
        """
        Args:
            user_pref: (B, D)
        Returns:
            augmented_pref: (B, D)
        """
        B = user_pref.shape[0]
        q = user_pref.unsqueeze(1)  # (B, 1, D)
        k = self.memory_key.unsqueeze(0).expand(B, -1, -1)
        v = self.memory_val.unsqueeze(0).expand(B, -1, -1)
        out, _ = self.attn(q, k, v)
        return user_pref + out.squeeze(1)


class QueryAgentR1(nn.Module):
    """
    Full QueryAgent-R1 model.

    Forward returns two types of outputs:
      - SL (supervised): cross-entropy on next-query prediction (L_CTR)
      - RL: REINFORCE on CVR reward from retrieved products (L_RL)

    Total loss: L = L_CTR + lambda_rl * L_RL
    """

    def __init__(
        self,
        n_queries: int,
        n_products: int,
        product_embs: torch.Tensor,    # (n_products, embed_dim) — retrieval index
        query_embed_dim: int = 64,
        hidden_dim: int = 128,
        memory_size: int = 16,
        lambda_rl: float = 0.5,
    ):
        super().__init__()
        self.lambda_rl = lambda_rl
        self.n_queries = n_queries

        self.encoder = QueryEncoder(n_queries, query_embed_dim, hidden_dim)
        self.memory = MemoryModule(hidden_dim, memory_size)
        self.gen_head = QueryProjectionHead(hidden_dim, n_queries)

        # Register product embeddings as buffer (retrieval index — frozen)
        self.register_buffer("product_embs", F.normalize(product_embs, dim=-1))

        # Reward-CVR scorer: scores user-product match
        self.reward_mlp = nn.Sequential(
            nn.Linear(hidden_dim + product_embs.shape[1], hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def retrieve(self, query_ids: torch.Tensor, top_k: int = 5) -> torch.Tensor:
        """
        Simulate retrieval: return top-k product embeddings for each query.
        Uses query embedding dot-product similarity.
        (In real system: dedicated retrieval engine called by Agent)
        """
        # We use the gen_head weight as query embeddings (simplified)
        # In real system: an actual retrieval engine returns product IDs/embs
        B = query_ids.shape[0]
        # Random retrieval for toy (real: ANN search)
        rand_idx = torch.randint(0, self.product_embs.shape[0], (B, top_k),
                                 device=query_ids.device)
        return self.product_embs[rand_idx]  # (B, top_k, prod_dim)

    def compute_cvr_reward(
        self,
        user_pref: torch.Tensor,       # (B, D)
        retrieved_prods: torch.Tensor, # (B, k, prod_dim)
    ) -> torch.Tensor:
        """
        Reward = mean CVR score of retrieved products for user.
        R = mean_k sigmoid(MLP([u; v_k]))   (Eq. 2)
        """
        B, k, prod_dim = retrieved_prods.shape
        u_expand = user_pref.unsqueeze(1).expand(-1, k, -1)  # (B, k, D)
        inp = torch.cat([u_expand, retrieved_prods], dim=-1)  # (B, k, D+prod_dim)
        scores = self.reward_mlp(inp).squeeze(-1)              # (B, k)
        reward = torch.sigmoid(scores).mean(dim=-1)            # (B,)
        return reward

    def forward(
        self,
        query_seq: torch.Tensor,         # (B, L)
        next_query: torch.Tensor,        # (B,) — ground-truth next query
        retrieved_prods: torch.Tensor,   # (B, 5, prod_dim) — pre-retrieved products
        cvr_signal: Optional[torch.Tensor] = None,  # (B,) — optional real CVR
    ) -> dict:
        # Encode user preference
        user_pref = self.encoder(query_seq)      # (B, D)
        user_pref = self.memory(user_pref)        # (B, D) — memory augmented

        # ── CTR loss (supervised next-query prediction) ─────────────────
        logits = self.gen_head(user_pref)         # (B, n_queries)
        L_CTR = F.cross_entropy(logits, next_query)  # Eq. 4

        # ── RL loss (REINFORCE with CVR reward) ─────────────────────────
        # Sample a query from the policy distribution
        with torch.no_grad():
            probs = F.softmax(logits.detach(), dim=-1)
            sampled_q = torch.multinomial(probs, 1).squeeze(-1)  # (B,)

        # Retrieve products for sampled query (agent action)
        retrieved_for_sampled = self.retrieve(sampled_q)  # (B, k, prod_dim)

        # Compute reward
        if cvr_signal is not None:
            reward = cvr_signal                             # use real CVR if available
        else:
            reward = self.compute_cvr_reward(user_pref, retrieved_for_sampled)  # Eq. 2

        # Normalize reward (baseline subtraction)
        reward = (reward - reward.mean()) / (reward.std() + 1e-8)

        # REINFORCE loss: -R * log P(sampled_q | seq)  (Eq. 3)
        log_probs = F.log_softmax(logits, dim=-1)
        sampled_log_prob = log_probs.gather(1, sampled_q.unsqueeze(1)).squeeze(1)  # (B,)
        L_RL = -(reward * sampled_log_prob).mean()

        # Total loss  (Eq. 5)
        loss = L_CTR + self.lambda_rl * L_RL

        return {
            "loss": loss,
            "L_CTR": L_CTR.item(),
            "L_RL": L_RL.item(),
            "logits": logits,
        }

    @torch.no_grad()
    def recommend_queries(
        self,
        query_seq: torch.Tensor,
        top_k: int = 10,
    ) -> torch.Tensor:
        """Return top-k recommended query IDs."""
        user_pref = self.encoder(query_seq)
        user_pref = self.memory(user_pref)
        logits = self.gen_head(user_pref)
        return logits.topk(top_k, dim=-1).indices


if __name__ == "__main__":
    from data import generate_toy_ecommerce_data
    data = generate_toy_ecommerce_data()

    product_embs = torch.FloatTensor(data["product_embs"])
    model = QueryAgentR1(
        n_queries=data["n_queries"],
        n_products=data["n_products"],
        product_embs=product_embs,
    )

    B = 8
    query_seq = torch.randint(1, data["n_queries"], (B, 10))
    next_query = torch.randint(0, data["n_queries"], (B,))
    retrieved_prods = torch.randn(B, 5, data["embed_dim"])
    cvr_signal = torch.rand(B)

    out = model(query_seq, next_query, retrieved_prods, cvr_signal)
    print(f"Loss: {out['loss'].item():.4f} | L_CTR: {out['L_CTR']:.4f} | L_RL: {out['L_RL']:.4f}")
    recs = model.recommend_queries(query_seq)
    print(f"Recommended queries shape: {recs.shape}")
    print("Model OK.")
