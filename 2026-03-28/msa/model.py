from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MSAConfig:
    vocab_size: int = 256
    d_model: int = 96
    n_docs: int = 32
    doc_len: int = 32
    chunk_size: int = 8
    top_k: int = 4


def _chunk_mean(x: torch.Tensor, chunk_size: int) -> torch.Tensor:
    # x: [B,N,L,D] -> [B,N,Lc,D]
    b, n, l, d = x.shape
    assert l % chunk_size == 0
    lc = l // chunk_size
    return x.view(b, n, lc, chunk_size, d).mean(dim=3)


class MSALayer(nn.Module):
    def __init__(self, cfg: MSAConfig):
        super().__init__()
        self.cfg = cfg
        self.wq = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        # routing projections
        self.wq_r = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wk_r = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, docs_h: torch.Tensor, query_h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # docs_h: [B,N,L,D], query_h: [B,Q,D]
        b, n, l, d = docs_h.shape

        k = self.wk(docs_h)
        v = self.wv(docs_h)
        k_r = self.wk_r(docs_h)

        k_bar = _chunk_mean(k, self.cfg.chunk_size)  # [B,N,Lc,D]
        v_bar = _chunk_mean(v, self.cfg.chunk_size)
        k_r_bar = _chunk_mean(k_r, self.cfg.chunk_size)

        q = self.wq(query_h)  # [B,Q,D]
        q_r = self.wq_r(query_h).mean(dim=1)  # [B,D]
        q_r = F.normalize(q_r, dim=-1)

        # routing scores per doc
        # score[b,n] = max over chunks of cos(q_r, k_r_bar)
        k_r_bar_n = F.normalize(k_r_bar, dim=-1)
        sim = torch.einsum("bd,bnld->bnl", q_r, k_r_bar_n)  # [B,N,Lc]
        doc_score = sim.max(dim=-1).values  # [B,N]

        topk = torch.topk(doc_score, k=min(self.cfg.top_k, n), dim=-1)
        idx = topk.indices  # [B,K]

        # gather selected docs and flatten chunks
        k_sel = torch.gather(k_bar, dim=1, index=idx[:, :, None, None].expand(b, idx.shape[1], k_bar.shape[2], d))
        v_sel = torch.gather(v_bar, dim=1, index=idx[:, :, None, None].expand(b, idx.shape[1], v_bar.shape[2], d))

        k_ctx = k_sel.reshape(b, -1, d)  # [B,K*Lc,D]
        v_ctx = v_sel.reshape(b, -1, d)

        # attend query to selected memory
        attn = torch.einsum("bqd,bkd->bqk", q, k_ctx) / (d**0.5)
        attn = attn.softmax(dim=-1)
        out = torch.einsum("bqk,bkd->bqd", attn, v_ctx)
        return self.out(out), doc_score


class MSAModel(nn.Module):
    def __init__(self, cfg: MSAConfig):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.msa = MSALayer(cfg)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, docs: torch.Tensor, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # docs: [B,N,L], query: [B,Q]
        docs_h = self.emb(docs)
        query_h = self.emb(query)
        out, doc_score = self.msa(docs_h, query_h)
        logits = self.lm_head(out[:, 0])  # use first query position as answer token
        return logits, doc_score
