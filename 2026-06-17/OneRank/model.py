from __future__ import annotations

"""OneRank — Transformer-native multi-task ranking architecture (toy reproduction).

Faithful to the paper's bottom-up forward pass:

  1. Structured Tokenization
       [ history tokens | preference-anchor tokens |
         per-candidate group: (candidate emb + 3 task tokens t_click/t_cart/t_order) ]
  2. Task-Specific Encoding (xL Transformer layers) with a STRUCTURED ATTENTION MASK:
       - the 3 task tokens inside a candidate group are MUTUALLY INVISIBLE,
       - candidate groups are ISOLATED from one another,
       - user-context (history+anchor) is CAUSALLY visible.
  3. Candidate-Aware Contextualization:
       a Situational Descriptor (pooled user context) is projected per-task into a
       query that does multi-head CROSS-attention over the per-task candidate tokens
       -> per-task global representation h_k.
  4. Multi-Task Prediction:
       cross-task relational attention over {h_k} WITH GRADIENT DETACHMENT
       (keys/values are detached so only diagonal/self gradients flow), then
       residual + LayerNorm + FFN -> z_k.
  5. Matching-based Scoring:
       s_k^i = <z_k, r_k^i>  (inner product per task and candidate).

Reference: Tang et al., "OneRank: Unified Transformer-Native Ranking Architecture
for Multi-Task Recommendation", arXiv:2606.16838.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

N_TASKS = 3  # click, cart, order

# token-type ids used by the structured tokenizer
T_HIST, T_ANCHOR, T_CAND, T_TASK = 0, 1, 2, 3


@dataclass(frozen=True)
class OneRankConfig:
    n_items: int = 200
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2        # L task-specific encoding layers
    ffn_dim: int = 128
    hist_len: int = 8
    anchor_len: int = 4
    n_candidates: int = 6
    dropout: float = 0.1


def build_structured_mask(cfg: OneRankConfig, device) -> torch.Tensor:
    """Return a boolean [S, S] attention mask (True = NOT allowed to attend).

    Layout of the sequence:
      user context : positions 0 .. U-1   (U = hist_len + anchor_len)
      candidate g  : base = U + g*4  ->  [cand, task_click, task_cart, task_order]
    """
    H, A, N = cfg.hist_len, cfg.anchor_len, cfg.n_candidates
    U = H + A
    S = U + N * (1 + N_TASKS)

    # start fully masked, then open the allowed pairs
    mask = torch.ones((S, S), dtype=torch.bool, device=device)

    def region(idx):
        if idx < U:
            return ("user", idx, 0)
        g = (idx - U) // (1 + N_TASKS)
        off = (idx - U) % (1 + N_TASKS)  # 0 = candidate token, 1..3 = task tokens
        return ("cand", g, off)

    for qi in range(S):
        rq = region(qi)
        for kj in range(S):
            rk = region(kj)
            allowed = False
            if rq[0] == "user":
                # user query: causal over user context only (candidate-agnostic)
                if rk[0] == "user" and kj <= qi:
                    allowed = True
            else:  # candidate-group query
                gq, oq = rq[1], rq[2]
                if rk[0] == "user":
                    allowed = True                      # candidates see all user context
                else:
                    gk, ok = rk[1], rk[2]
                    if gk == gq:                        # same candidate group only
                        both_task = oq >= 1 and ok >= 1
                        if both_task and oq != ok:
                            allowed = False             # task tokens mutually invisible
                        else:
                            allowed = True
                    # different candidate group -> isolated (stays masked)
            if allowed:
                mask[qi, kj] = False
    return mask


class EncoderLayer(nn.Module):
    """Pre-norm Transformer encoder layer accepting a fixed [S, S] attn mask."""

    def __init__(self, cfg: OneRankConfig):
        super().__init__()
        self.attn = nn.MultiheadAttention(cfg.d_model, cfg.n_heads,
                                          dropout=cfg.dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.ffn_dim), nn.GELU(),
            nn.Dropout(cfg.dropout), nn.Linear(cfg.ffn_dim, cfg.d_model),
        )

    def forward(self, x, attn_mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + a
        x = x + self.ffn(self.ln2(x))
        return x


class OneRank(nn.Module):
    def __init__(self, cfg: OneRankConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        # ---- (1) tokenization embeddings -----------------------------------
        self.item_emb = nn.Embedding(cfg.n_items, d)          # history/anchor/candidate items
        self.type_emb = nn.Embedding(4, d)                    # hist / anchor / cand / task
        self.task_emb = nn.Embedding(N_TASKS, d)              # identity of the 3 task tokens
        max_ctx = cfg.hist_len + cfg.anchor_len
        self.pos_emb = nn.Embedding(max_ctx, d)               # positions for user context
        self.dropout = nn.Dropout(cfg.dropout)

        # ---- (2) task-specific encoding stack ------------------------------
        self.layers = nn.ModuleList([EncoderLayer(cfg) for _ in range(cfg.n_layers)])

        # ---- (3) candidate-aware contextualization -------------------------
        # per-task query projection of the situational descriptor
        self.q_proj = nn.ModuleList([nn.Linear(d, d) for _ in range(N_TASKS)])
        self.ctx_attn = nn.MultiheadAttention(d, cfg.n_heads, dropout=cfg.dropout,
                                              batch_first=True)

        # ---- (4) cross-task relational attention (read-only memory) --------
        self.cross_attn = nn.MultiheadAttention(d, cfg.n_heads, dropout=cfg.dropout,
                                                batch_first=True)
        self.ln_x1 = nn.LayerNorm(d)
        self.ln_x2 = nn.LayerNorm(d)
        self.x_ffn = nn.Sequential(
            nn.Linear(d, cfg.ffn_dim), nn.GELU(), nn.Linear(cfg.ffn_dim, d),
        )

        # ---- (5) learnable temperature for matching-based scoring ----------
        # keeps the inner-product logits in a useful range for BCE + softmax.
        self.logit_scale = nn.Parameter(torch.tensor(float(d) ** -0.5).log())

        self.register_buffer("attn_mask", build_structured_mask(cfg, "cpu"), persistent=False)

    # ------------------------------------------------------------------ #
    def tokenize(self, hist, anchor, cands):
        """(1) Structured tokenization -> [B, S, d] token embeddings."""
        cfg = self.cfg
        B = hist.size(0)
        device = hist.device

        # user context: history + anchor, with type + positional embeddings
        ctx_ids = torch.cat([hist, anchor], dim=1)                 # [B, U]
        U = ctx_ids.size(1)
        ctx_type = torch.cat([
            torch.full((B, cfg.hist_len), T_HIST, device=device),
            torch.full((B, cfg.anchor_len), T_ANCHOR, device=device),
        ], dim=1)
        pos = torch.arange(U, device=device).unsqueeze(0).expand(B, -1)
        ctx_tok = self.item_emb(ctx_ids) + self.type_emb(ctx_type) + self.pos_emb(pos)

        # per-candidate group: (candidate emb + 3 task tokens)
        cand_tok = self.item_emb(cands) + self.type_emb(
            torch.full_like(cands, T_CAND))                        # [B, N, d]
        task_base = self.task_emb(torch.arange(N_TASKS, device=device)) \
            + self.type_emb(torch.full((N_TASKS,), T_TASK, device=device))   # [3, d]

        groups = []
        for g in range(cfg.n_candidates):
            c = cand_tok[:, g, :].unsqueeze(1)                     # [B, 1, d]
            t = task_base.unsqueeze(0).expand(B, -1, -1)           # [B, 3, d]
            groups.append(torch.cat([c, t], dim=1))                # [B, 4, d]
        groups = torch.cat(groups, dim=1)                          # [B, N*4, d]

        tokens = torch.cat([ctx_tok, groups], dim=1)               # [B, S, d]
        return self.dropout(tokens), U

    def forward(self, hist, anchor, cands):
        cfg = self.cfg
        B = hist.size(0)
        x, U = self.tokenize(hist, anchor, cands)
        mask = self.attn_mask.to(x.device)

        # ---- (2) task-specific encoding ------------------------------------
        for layer in self.layers:
            x = layer(x, mask)                                     # [B, S, d]

        # gather per-task candidate token reps: r_k^i  (encoded task tokens)
        # candidate group g occupies positions U + g*4 + {0:cand,1:click,2:cart,3:order}
        task_reps = []   # list over tasks of [B, N, d]
        for k in range(N_TASKS):
            idxs = [U + g * (1 + N_TASKS) + (1 + k) for g in range(cfg.n_candidates)]
            task_reps.append(x[:, idxs, :])                        # [B, N, d]
        T = torch.stack(task_reps, dim=1)                          # [B, 3, N, d]

        # ---- (3) candidate-aware contextualization -------------------------
        sd = x[:, :U, :].mean(dim=1, keepdim=True)                 # situational descriptor [B,1,d]
        h_list = []
        for k in range(N_TASKS):
            q = self.q_proj[k](sd)                                 # [B, 1, d] per-task query
            kv = task_reps[k]                                      # [B, N, d]
            h_k, _ = self.ctx_attn(q, kv, kv, need_weights=False)  # MHCA over candidates
            h_list.append(h_k)                                     # [B, 1, d]
        h = torch.cat(h_list, dim=1)                               # [B, 3, d]

        # ---- (4) multi-task prediction w/ GRADIENT DETACHMENT --------------
        # keys/values are detached -> cross-task attention is a *read-only* memory;
        # gradients flow only through the query (self/diagonal path) + residual.
        kv = h.detach()
        o, _ = self.cross_attn(h, kv, kv, need_weights=False)      # [B, 3, d]
        z = self.ln_x1(h + o)
        z = self.ln_x2(z + self.x_ffn(z))                          # [B, 3, d]

        # ---- (5) matching-based scoring  s_k^i = <z_k, r_k^i> --------------
        scores = torch.einsum("bkd,bknd->bkn", z, T)              # [B, 3, N]
        scores = scores * self.logit_scale.exp()                  # learnable temperature
        return scores
