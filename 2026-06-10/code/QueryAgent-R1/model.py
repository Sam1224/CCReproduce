"""
QueryAgent-R1 model reproduction.

Architecture (faithful to paper §3):
  1. MemoryEncoder    — extracts an interest graph from user long-term history
  2. QueryPolicyLM    — generates candidate queries conditioned on memory + context
  3. RetrievalModule  — retrieves products for a generated query (dense search)
  4. QueryAgent       — orchestrates the chain-of-retrieval loop

The RL training objective is a consistency reward:
  r(q) = α · CTR_proxy(q) + β · CVR_proxy(q, retrieved_products)

where CTR_proxy = query–history similarity and
      CVR_proxy = alignment between retrieved products and purchase history.

Formula (paper eq. 5):
  R_consistency(q, P) = sim(E_q, E_history) * sim(E_P, E_purchase)

where E_q, E_history are mean pooled query / history embeddings,
      E_P is mean pooled retrieved product embeddings,
      E_purchase is mean pooled purchase history embeddings.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional

from data import Product, UserHistory, ProductCatalog


# ---------------------------------------------------------------------------
# Memory Encoder — Interest Graph
# ---------------------------------------------------------------------------

class InterestGraphEncoder(nn.Module):
    """
    Encodes user long-term history into an 'interest graph' representation.

    Simplified from paper: we use a 2-layer Transformer over item embeddings
    (both clicked and purchased), weighted by recency and interaction type.
    The output is a single context vector used to condition query generation.

    Paper §3.2: "memory abstraction module extracts an interest graph
    from users' long-term memory for efficient user profiling."
    """

    def __init__(self, item_emb_dim: int, hidden_dim: int, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.item_proj = nn.Linear(item_emb_dim, hidden_dim)
        self.type_emb = nn.Embedding(3, hidden_dim)  # 0=pad, 1=click, 2=purchase
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 2,
            batch_first=True, dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pool = nn.Linear(hidden_dim, hidden_dim)  # weighted pooling gate

    def forward(
        self,
        item_embeddings: torch.Tensor,  # (B, seq_len, item_emb_dim)
        interaction_types: torch.Tensor,  # (B, seq_len) int, 1=click 2=purchase
        attention_mask: Optional[torch.Tensor] = None,  # (B, seq_len) bool
    ) -> torch.Tensor:
        """Returns interest graph context vector of shape (B, hidden_dim)."""
        x = self.item_proj(item_embeddings) + self.type_emb(interaction_types)
        # Transformer (causal=False: full attention over history)
        if attention_mask is not None:
            key_padding_mask = ~attention_mask  # True = ignore
        else:
            key_padding_mask = None
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)  # (B, L, H)
        # Weighted mean pooling
        gate = torch.sigmoid(self.pool(x))  # (B, L, H)
        if attention_mask is not None:
            gate = gate * attention_mask.unsqueeze(-1).float()
        context = (gate * x).sum(dim=1) / (gate.sum(dim=1) + 1e-8)  # (B, H)
        return context


# ---------------------------------------------------------------------------
# Query Policy LM — lightweight transformer-based query generator
# ---------------------------------------------------------------------------

class QueryPolicyLM(nn.Module):
    """
    Lightweight autoregressive query generator conditioned on:
      - user interest context (from MemoryEncoder)
      - retrieved product feedback (chain-of-retrieval signal)

    In production this would be a fine-tuned LLM (e.g., Qwen/LLaMA).
    Here we use a small transformer over a toy vocabulary.

    Paper §3.3: "QueryAgent-R1 grounds query generation in real inventory
    retrieval, allowing the agent to validate and refine queries based on
    retrieved products."
    """

    VOCAB_SIZE = 512
    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2

    def __init__(self, hidden_dim: int, n_layers: int = 3, n_heads: int = 4, max_len: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_len = max_len

        self.token_emb = nn.Embedding(self.VOCAB_SIZE, hidden_dim, padding_idx=self.PAD_ID)
        self.pos_emb = nn.Embedding(max_len + 2, hidden_dim)
        self.context_proj = nn.Linear(hidden_dim, hidden_dim)  # maps user context

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 2,
            batch_first=True, dropout=0.1,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.lm_head = nn.Linear(hidden_dim, self.VOCAB_SIZE)

    def forward(
        self,
        tgt_ids: torch.Tensor,          # (B, T)
        user_context: torch.Tensor,     # (B, H)
        retrieval_feedback: Optional[torch.Tensor] = None,  # (B, H) mean of retrieved embeddings
    ) -> torch.Tensor:
        """Returns logits of shape (B, T, VOCAB_SIZE)."""
        B, T = tgt_ids.shape
        positions = torch.arange(T, device=tgt_ids.device).unsqueeze(0)
        x = self.token_emb(tgt_ids) + self.pos_emb(positions)

        # Build memory from user context + optional retrieval feedback
        mem = self.context_proj(user_context).unsqueeze(1)  # (B, 1, H)
        if retrieval_feedback is not None:
            rf = retrieval_feedback.unsqueeze(1)  # (B, 1, H)
            mem = torch.cat([mem, rf], dim=1)  # (B, 2, H)

        # Causal mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T, device=tgt_ids.device)
        x = self.decoder(x, mem, tgt_mask=tgt_mask)  # (B, T, H)
        return self.lm_head(x)  # (B, T, VOCAB_SIZE)

    @torch.no_grad()
    def generate(
        self,
        user_context: torch.Tensor,  # (B, H)
        retrieval_feedback: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        max_new_tokens: int = 16,
    ) -> torch.Tensor:
        """Greedy / temperature-sampled generation. Returns (B, T) token ids."""
        B = user_context.shape[0]
        device = user_context.device
        ids = torch.full((B, 1), self.BOS_ID, dtype=torch.long, device=device)
        for _ in range(max_new_tokens):
            logits = self.forward(ids, user_context, retrieval_feedback)[:, -1, :]
            if temperature != 1.0:
                logits = logits / temperature
            next_id = torch.multinomial(F.softmax(logits, dim=-1), 1)
            ids = torch.cat([ids, next_id], dim=1)
            if (next_id == self.EOS_ID).all():
                break
        return ids


# ---------------------------------------------------------------------------
# Retrieval Module — dense product search
# ---------------------------------------------------------------------------

class RetrievalModule(nn.Module):
    """
    Encodes a query (as token ids) into an embedding and retrieves
    top-k products from the product catalog (pure numpy nearest-neighbor).

    Paper §3.3: "chain-of-retrieval optimization — the agent validates and
    refines queries based on retrieved products."
    """

    def __init__(self, vocab_size: int, hidden_dim: int, catalog_dim: int):
        super().__init__()
        self.query_encoder = nn.EmbeddingBag(vocab_size, hidden_dim, mode="mean")
        self.proj = nn.Linear(hidden_dim, catalog_dim)  # project to catalog space

    def encode_query(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids: (B, T) → query embedding (B, catalog_dim)."""
        h = self.query_encoder(token_ids)  # (B, hidden_dim)
        return F.normalize(self.proj(h), dim=-1)  # (B, catalog_dim)

    def retrieve(
        self,
        query_emb: np.ndarray,   # (B, catalog_dim) numpy
        catalog: ProductCatalog,
        top_k: int = 5,
    ) -> List[List[Product]]:
        results = []
        for q in query_emb:
            results.append(catalog.search(q, top_k=top_k))
        return results


# ---------------------------------------------------------------------------
# Consistency Reward
# ---------------------------------------------------------------------------

def consistency_reward(
    query_emb: torch.Tensor,           # (B, D) — generated query embedding
    history_emb: torch.Tensor,         # (B, D) — user history mean embedding
    retrieved_emb: torch.Tensor,       # (B, D) — mean retrieved product embedding
    purchase_emb: torch.Tensor,        # (B, D) — user purchase history mean embedding
    alpha: float = 0.5,
    beta: float = 0.5,
) -> torch.Tensor:
    """
    Paper eq. (5):
      R = α * sim(q, history) + β * sim(retrieved, purchase)

    Both similarity terms are cosine similarity in [−1, 1].
    Returns reward tensor of shape (B,).
    """
    ctr_proxy = F.cosine_similarity(query_emb, history_emb, dim=-1)
    cvr_proxy = F.cosine_similarity(retrieved_emb, purchase_emb, dim=-1)
    return alpha * ctr_proxy + beta * cvr_proxy


# ---------------------------------------------------------------------------
# Full QueryAgent-R1 Model
# ---------------------------------------------------------------------------

class QueryAgentR1(nn.Module):
    """
    Full QueryAgent-R1:
      - Encodes user history → interest context
      - Generates candidate query
      - Retrieves products → computes feedback embedding
      - Optionally refines query (chain-of-retrieval)
      - Computes consistency reward for RL training

    Training uses GRPO-style group reward (paper §3.4).
    """

    def __init__(
        self,
        catalog_emb_dim: int = 64,
        hidden_dim: int = 128,
        vocab_size: int = QueryPolicyLM.VOCAB_SIZE,
        n_refinement_steps: int = 2,
    ):
        super().__init__()
        self.n_refinement_steps = n_refinement_steps

        self.memory_encoder = InterestGraphEncoder(
            item_emb_dim=catalog_emb_dim,
            hidden_dim=hidden_dim,
        )
        self.policy_lm = QueryPolicyLM(
            hidden_dim=hidden_dim,
            max_len=32,
        )
        self.retrieval_module = RetrievalModule(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            catalog_dim=catalog_emb_dim,
        )

        # Project retrieved product embeddings to policy space
        self.prod_proj = nn.Linear(catalog_emb_dim, hidden_dim)

    def encode_history(
        self,
        clicked_embs: torch.Tensor,    # (B, L_click, D)
        purchased_embs: torch.Tensor,  # (B, L_purch, D)
        click_mask: Optional[torch.Tensor] = None,
        purch_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          context   (B, H)  — user interest context
          hist_emb  (B, D)  — mean clicked embedding (for reward)
          purch_emb (B, D)  — mean purchased embedding (for reward)
        """
        B = clicked_embs.shape[0]
        # Combine clicked + purchased for the memory encoder
        if click_mask is None:
            click_mask = torch.ones(clicked_embs.shape[:2], dtype=torch.bool,
                                    device=clicked_embs.device)
        if purch_mask is None:
            purch_mask = torch.ones(purchased_embs.shape[:2], dtype=torch.bool,
                                    device=purchased_embs.device)

        # Interaction type: 1=click, 2=purchase
        click_types = torch.ones(clicked_embs.shape[:2], dtype=torch.long,
                                  device=clicked_embs.device)
        purch_types = torch.full(purchased_embs.shape[:2], 2, dtype=torch.long,
                                  device=purchased_embs.device)

        all_embs = torch.cat([clicked_embs, purchased_embs], dim=1)
        all_types = torch.cat([click_types, purch_types], dim=1)
        all_mask = torch.cat([click_mask, purch_mask], dim=1)

        context = self.memory_encoder(all_embs, all_types, all_mask)  # (B, H)

        # Mean embeddings for reward computation
        hist_emb = (clicked_embs * click_mask.unsqueeze(-1).float()).sum(1) \
                   / (click_mask.float().sum(1, keepdim=True) + 1e-8)  # (B, D)
        purch_emb = (purchased_embs * purch_mask.unsqueeze(-1).float()).sum(1) \
                    / (purch_mask.float().sum(1, keepdim=True) + 1e-8)  # (B, D)

        return context, hist_emb, purch_emb

    def generate_and_retrieve(
        self,
        user_context: torch.Tensor,  # (B, H)
        catalog: ProductCatalog,
        retrieval_feedback: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[Product]]]:
        """
        One step of chain-of-retrieval:
          1. Generate query tokens
          2. Encode query → dense embedding
          3. Retrieve products from catalog
          4. Compute mean retrieved embedding (for next step feedback)
        Returns (query_ids, query_emb, retrieved_products)
        """
        # Generate query
        query_ids = self.policy_lm.generate(
            user_context, retrieval_feedback, temperature=temperature
        )  # (B, T)

        # Encode generated query
        query_emb = self.retrieval_module.encode_query(query_ids)  # (B, catalog_dim)
        query_emb_np = query_emb.detach().cpu().numpy()

        # Retrieve
        retrieved = self.retrieval_module.retrieve(query_emb_np, catalog, top_k=5)

        # Mean retrieved embedding for feedback
        # (In training, embeddings come from catalog._embeddings; here we look them up)
        ret_emb_list = []
        for batch_results in retrieved:
            embs = np.stack([p.embedding for p in batch_results])  # (k, D)
            ret_emb_list.append(embs.mean(0))
        ret_emb_np = np.stack(ret_emb_list)  # (B, D)
        ret_emb = torch.tensor(ret_emb_np, dtype=torch.float32, device=user_context.device)

        # Project retrieved embeddings to policy hidden space for feedback
        ret_feedback = self.prod_proj(ret_emb)  # (B, H)

        return query_ids, query_emb, retrieved, ret_feedback

    def forward(
        self,
        clicked_embs: torch.Tensor,
        purchased_embs: torch.Tensor,
        catalog: ProductCatalog,
        temperature: float = 1.2,
    ) -> dict:
        """
        Full chain-of-retrieval forward pass. Returns dict with:
          - query_ids_final: final generated query tokens
          - query_emb_final: final query embedding
          - retrieved_products: final retrieved products
          - reward: consistency reward (scalar per sample in batch)
        """
        context, hist_emb, purch_emb = self.encode_history(
            clicked_embs, purchased_embs
        )

        feedback = None
        query_ids, query_emb, retrieved, feedback = self.generate_and_retrieve(
            context, catalog, feedback, temperature
        )

        # Chain-of-retrieval refinement steps
        for _ in range(self.n_refinement_steps - 1):
            query_ids, query_emb, retrieved, feedback = self.generate_and_retrieve(
                context, catalog, feedback, temperature
            )

        # Compute consistency reward
        # Mean retrieved product embedding
        ret_emb_list = []
        for batch_results in retrieved:
            embs = np.stack([p.embedding for p in batch_results])
            ret_emb_list.append(embs.mean(0))
        ret_emb = torch.tensor(
            np.stack(ret_emb_list), dtype=torch.float32, device=query_emb.device
        )

        reward = consistency_reward(
            query_emb, hist_emb, ret_emb, purch_emb
        )

        return {
            "query_ids_final": query_ids,
            "query_emb_final": query_emb,
            "retrieved_products": retrieved,
            "reward": reward,
            "context": context,
        }


if __name__ == "__main__":
    from data import build_datasets

    catalog, users, train_set, eval_set = build_datasets(n_products=200, n_users=50)

    model = QueryAgentR1(catalog_emb_dim=64, hidden_dim=64)
    model.eval()

    B = 4
    click_emb = torch.randn(B, 10, 64)
    purch_emb = torch.randn(B, 3, 64)

    with torch.no_grad():
        out = model(click_emb, purch_emb, catalog)

    print(f"Generated query ids shape: {out['query_ids_final'].shape}")
    print(f"Query embedding shape: {out['query_emb_final'].shape}")
    print(f"Reward: {out['reward']}")
    print(f"Retrieved product sample: {out['retrieved_products'][0][0].title}")
