from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    num_items: int
    num_requests: int
    d_model: int = 96
    nhead: int = 4
    nlayers: int = 2
    dropout: float = 0.1

    adapter_dim: int = 24
    adapter_scale: float = 1.0


class ResidualAdapter(nn.Module):
    """Bottleneck residual adapter: x <- x + s * Up(Act(Down(LN(x))))."""

    def __init__(self, d_model: int, adapter_dim: int, scale: float = 1.0) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.down = nn.Linear(d_model, adapter_dim)
        self.up = nn.Linear(adapter_dim, d_model)
        self.scale = scale
        # initialize close to zero so inserting adapters doesn't disrupt pretrained backbone
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        h = F.gelu(self.down(h))
        h = self.up(h)
        return x + self.scale * h


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float,
        adapter_dim: int,
        adapter_scale: float,
        use_adapter: bool,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

        self.use_adapter = use_adapter
        self.adapter = ResidualAdapter(d_model, adapter_dim, scale=adapter_scale) if use_adapter else None

    def forward(
        self,
        x: torch.Tensor,
        *,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.drop1(a)

        h = self.ln2(x)
        x = x + self.ffn(h)

        if self.adapter is not None:
            x = self.adapter(x)
        return x


class LazFormer(nn.Module):
    """Toy LazFormer: generative pretraining + adapters + request-aware sparse attention ranking."""

    def __init__(
        self,
        cfg: ModelConfig,
        *,
        use_adapter: bool = True,
        topm_history: int = 12,
        use_sparse: bool = True,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_items = cfg.num_items
        self.num_requests = cfg.num_requests
        self.topm_history = topm_history
        self.use_sparse = use_sparse

        self.item_emb = nn.Embedding(cfg.num_items + 1, cfg.d_model)  # include PAD=0
        self.req_emb = nn.Embedding(cfg.num_requests, cfg.d_model)
        self.pos_emb = nn.Embedding(1 + 512, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    cfg.d_model,
                    cfg.nhead,
                    cfg.dropout,
                    cfg.adapter_dim,
                    cfg.adapter_scale,
                    use_adapter=use_adapter,
                )
                for _ in range(cfg.nlayers)
            ]
        )
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # pretrain head: predict next item id
        self.lm_head = nn.Linear(cfg.d_model, cfg.num_items + 1)

        # rank head: score each candidate conditioned on request token hidden
        self.rank_head = nn.Sequential(
            nn.Linear(2 * cfg.d_model, cfg.d_model),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.d_model, 1),
        )

        # request-history similarity for coarse selection
        self.req_hist_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def _add_pos(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        return x + self.pos_emb(pos)

    def _run_blocks(
        self,
        x: torch.Tensor,
        *,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x = self.drop(self._add_pos(x))
        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        return self.ln_f(x)

    # ----------------------- generative pretraining -----------------------
    def pretrain_logits(self, req: torch.Tensor, seq: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (logits, valid_mask).

        Tokens: [REQ] + item_1..item_N (padded).
        Prediction positions: token_pos t predicts item_{t+1}.
        - request token (t=0) predicts item_1
        - item_t (t>=1) predicts item_{t+1}

        valid_mask marks positions where a target exists (t < N).
        """
        B, L = seq.shape
        req_tok = self.req_emb(req).unsqueeze(1)  # [B,1,D]
        item_tok = self.item_emb(seq)  # [B,L,D]
        x = torch.cat([req_tok, item_tok], dim=1)  # [B,1+L,D]

        T = x.shape[1]
        # causal mask (2D, shared across batch)
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)

        # key padding mask: True means PAD (masked out)
        # request token never padded
        item_pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        item_valid = item_pos < lengths.unsqueeze(1)
        kpm = torch.zeros(B, 1 + L, dtype=torch.bool, device=x.device)
        kpm[:, 1:] = ~item_valid

        h = self._run_blocks(x, attn_mask=causal, key_padding_mask=kpm)
        logits = self.lm_head(h)  # [B,T,V]

        # valid prediction positions t < N (lengths)
        tok_pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        valid = tok_pos < lengths.unsqueeze(1)
        return logits, valid

    def pretrain_loss(self, req: torch.Tensor, seq: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        logits, valid = self.pretrain_logits(req, seq, lengths)
        # targets align with token positions: target[t] = seq[t] for t < N
        B, L = seq.shape
        T = 1 + L
        targets = torch.zeros(B, T, dtype=torch.long, device=seq.device)
        targets[:, :L] = seq
        loss = F.cross_entropy(logits[valid], targets[valid])
        return loss

    # -------------------------- request-aware ranking --------------------------
    def _coarse_select_history(self, req: torch.Tensor, hist: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Return keep_mask [B,Lh] for coarse-to-fine compression.

        We keep:
        - Top-M history items by request-item similarity (request-aware), and
        - A small recent tail (recency prior, avoids missing near-end anchors).
        """
        B, Lh = hist.shape
        h_emb = self.item_emb(hist)  # [B,Lh,D]
        q = self.req_hist_proj(self.req_emb(req)).unsqueeze(1)  # [B,1,D]
        sim = (h_emb * q).sum(-1)  # [B,Lh]

        pos = torch.arange(Lh, device=hist.device).unsqueeze(0).expand(B, Lh)
        valid = pos < lengths.unsqueeze(1)
        sim = sim.masked_fill(~valid, float("-inf"))

        M = min(self.topm_history, Lh)
        top_idx = sim.topk(M, dim=1).indices  # [B,M]
        keep = torch.zeros(B, Lh, dtype=torch.bool, device=hist.device)
        keep.scatter_(1, top_idx, True)

        tail = min(6, Lh)
        tail_mask = pos >= (lengths.unsqueeze(1) - tail)
        keep = (keep | tail_mask) & valid
        return keep

    def _build_sparse_attn_mask(
        self,
        keep_hist: torch.Tensor,
        *,
        Lh: int,
        K: int,
    ) -> torch.Tensor:
        """Build a 3D bool mask for MultiheadAttention.

        Return attn_mask with shape [B*nhead, T, T], where True means "disallow".

        Layout: [REQ] + history(Lh) + candidates(K)

        - Candidates can attend to: REQ + selected history + all candidates
        - Selected history can attend to: REQ + selected history
        - Unselected history can attend to: REQ + itself
        - REQ can attend to: all
        """
        B = keep_hist.shape[0]
        T = 1 + Lh + K
        allow = torch.zeros(B, T, T, dtype=torch.bool, device=keep_hist.device)

        req_pos = 0
        hist_start = 1
        cand_start = 1 + Lh

        # REQ attends to all
        allow[:, req_pos, :] = True

        # precompute selected hist positions in token space
        sel_hist_tok = torch.zeros(B, T, dtype=torch.bool, device=keep_hist.device)
        sel_hist_tok[:, hist_start : hist_start + Lh] = keep_hist

        # history queries
        # selected: allow to REQ + selected hist
        # unselected: allow to REQ + itself
        for i in range(Lh):
            tok_i = hist_start + i
            is_sel = keep_hist[:, i]
            # everyone can attend to REQ
            allow[:, tok_i, req_pos] = True
            # selected attends to selected hist
            allow[is_sel, tok_i, :] |= sel_hist_tok[is_sel]
            # unselected attends to itself
            allow[~is_sel, tok_i, tok_i] = True

        # candidate queries: allow to REQ + selected history + all candidates
        allow[:, cand_start:, req_pos] = True
        allow[:, cand_start:, :] |= sel_hist_tok.unsqueeze(1).expand(B, K, T)
        allow[:, cand_start:, cand_start:] = True

        disallow = ~allow
        # expand over heads
        disallow = disallow.unsqueeze(1).expand(B, self.cfg.nhead, T, T).reshape(B * self.cfg.nhead, T, T)
        return disallow

    def rank_logits(self, req: torch.Tensor, hist: torch.Tensor, lengths: torch.Tensor, cands: torch.Tensor) -> torch.Tensor:
        """Return candidate logits [B,K]."""
        B, Lh = hist.shape
        K = cands.shape[1]

        req_tok = self.req_emb(req).unsqueeze(1)
        hist_tok = self.item_emb(hist)
        cand_tok = self.item_emb(cands)
        x = torch.cat([req_tok, hist_tok, cand_tok], dim=1)  # [B,1+Lh+K,D]

        # padding for history tokens only
        pos = torch.arange(Lh, device=hist.device).unsqueeze(0).expand(B, Lh)
        hist_valid = pos < lengths.unsqueeze(1)
        kpm = torch.zeros(B, 1 + Lh + K, dtype=torch.bool, device=hist.device)
        kpm[:, 1 : 1 + Lh] = ~hist_valid

        attn_mask = None
        if self.use_sparse:
            keep = self._coarse_select_history(req, hist, lengths)
            attn_mask = self._build_sparse_attn_mask(keep, Lh=Lh, K=K)

        h = self._run_blocks(x, attn_mask=attn_mask, key_padding_mask=kpm)

        h_req = h[:, 0]  # [B,D]
        h_c = h[:, 1 + Lh : 1 + Lh + K]  # [B,K,D]
        h_req_rep = h_req.unsqueeze(1).expand_as(h_c)
        feat = torch.cat([h_c, h_req_rep], dim=-1)
        logits = self.rank_head(feat).squeeze(-1)
        return logits

    def freeze_backbone_for_rank(self) -> None:
        """Freeze Transformer blocks, but keep lightweight parts trainable.

        Trainable in rank stage:
        - residual adapters
        - rank head
        - request embedding
        - item embedding (lets the model adapt item geometry to the target domain)
        - req->hist projection used in coarse selection
        - positional embedding

        Frozen:
        - attention / FFN weights in Transformer blocks
        """
        for name, p in self.named_parameters():
            if (
                ("adapter" in name)
                or ("rank_head" in name)
                or ("req_emb" in name)
                or ("item_emb" in name)
                or ("pos_emb" in name)
                or ("req_hist_proj" in name)
            ):
                p.requires_grad = True
            else:
                p.requires_grad = False


class BaselineRanker(nn.Module):
    """A simple baseline: mean(history) + request -> dot with candidate embeddings."""

    def __init__(self, num_items: int, num_requests: int, d: int = 64) -> None:
        super().__init__()
        self.item_emb = nn.Embedding(num_items + 1, d)
        self.req_emb = nn.Embedding(num_requests, d)
        self.fuse = nn.Linear(2 * d, d)

    def rank_logits(self, req: torch.Tensor, hist: torch.Tensor, lengths: torch.Tensor, cands: torch.Tensor) -> torch.Tensor:
        B, Lh = hist.shape
        K = cands.shape[1]

        h = self.item_emb(hist)  # [B,Lh,D]
        pos = torch.arange(Lh, device=hist.device).unsqueeze(0).expand(B, Lh)
        valid = pos < lengths.unsqueeze(1)
        h = h * valid.unsqueeze(-1)
        denom = valid.sum(1).clamp_min(1).unsqueeze(1)
        user = h.sum(1) / denom  # [B,D]

        q = self.req_emb(req)
        z = self.fuse(torch.cat([user, q], dim=-1))

        c = self.item_emb(cands)  # [B,K,D]
        logits = (c * z.unsqueeze(1)).sum(-1)
        return logits
