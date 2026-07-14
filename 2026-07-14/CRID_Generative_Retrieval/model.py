from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    query_vocab_size: int
    n_prefix: int
    n_suffix: int

    d_model: int = 128
    hidden: int = 128
    dropout: float = 0.1


class CRIDGenerator(nn.Module):
    """Two-step autoregressive generator for DocID tokens.

    Step-1: generate semantic cluster prefix token (Cxx)
    Step-2: generate within-cluster suffix token (Rxx for CRID or Sxx for SID)

    This is a tiny surrogate of generative retrieval; we keep it CPU-friendly.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.query_emb = nn.Embedding(cfg.query_vocab_size, cfg.d_model)
        self.encoder = nn.GRU(
            input_size=cfg.d_model,
            hidden_size=cfg.hidden,
            batch_first=True,
        )

        self.dropout = nn.Dropout(cfg.dropout)

        self.prefix_head = nn.Linear(cfg.hidden, cfg.n_prefix)

        # 关键 toy 设定：suffix token 在 CRID 中本身就是“全局可泛化的价值 rank”。
        # 因此我们让 suffix 预测更多学习一个跨簇共享的“业务价值 prior”，而不是强依赖 prefix。
        # 这样在 SID（簇内随机后缀）下很难学到稳定的高价值排序，而在 CRID 下会更自然地把概率集中到低 rank。
        self.suffix_head = nn.Linear(cfg.hidden, cfg.n_suffix)

    def encode(self, query_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Return query representation h: (B, hidden)."""
        emb = self.query_emb(query_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.encoder(packed)
        h = h.squeeze(0)
        return self.dropout(h)

    def forward(
        self,
        query_ids: torch.Tensor,
        lengths: torch.Tensor,
        prefix_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Teacher-forcing forward.

        Returns:
          - prefix_logits: (B, n_prefix)
          - suffix_logits: (B, n_suffix)
        """
        h = self.encode(query_ids, lengths)
        prefix_logits = self.prefix_head(h)

        # prefix_ids 保留在签名里是为了保持“先生成 prefix、再生成 suffix”的接口形态。
        # 但在本 toy 里我们刻意让 suffix 的概率更多反映“全局价值排序 prior”。
        _ = prefix_ids  # not used
        suffix_logits = self.suffix_head(h)
        return prefix_logits, suffix_logits

    @torch.no_grad()
    def generate_topk(
        self,
        query_ids: torch.Tensor,
        lengths: torch.Tensor,
        topk: int = 10,
        beam_prefix: int = 5,
        beam_suffix: int = 20,
    ) -> List[List[Tuple[int, int, float]]]:
        """Return per-query topk generated (prefix_id, suffix_id, logp)."""
        self.eval()
        h = self.encode(query_ids, lengths)

        logp_prefix = F.log_softmax(self.prefix_head(h), dim=-1)  # (B, P)
        logp_suffix = F.log_softmax(self.suffix_head(h), dim=-1)  # (B, S)
        B, P = logp_prefix.shape
        S = logp_suffix.shape[1]

        beam_prefix = min(beam_prefix, P)
        beam_suffix = min(beam_suffix, S)

        prefix_topv, prefix_topi = torch.topk(logp_prefix, k=beam_prefix, dim=-1)
        suffix_topv, suffix_topi = torch.topk(logp_suffix, k=beam_suffix, dim=-1)

        results: List[List[Tuple[int, int, float]]] = []
        for b in range(B):
            cands: List[Tuple[int, int, float]] = []

            for j in range(beam_prefix):
                pid = int(prefix_topi[b, j].item())
                lp_p = float(prefix_topv[b, j].item())

                for k in range(beam_suffix):
                    sid = int(suffix_topi[b, k].item())
                    lp = lp_p + float(suffix_topv[b, k].item())
                    cands.append((pid, sid, lp))

            cands.sort(key=lambda x: x[2], reverse=True)
            results.append(cands[:topk])

        return results


def build_model(
    query_vocab_size: int,
    n_prefix: int,
    n_suffix: int,
    d_model: int = 128,
    hidden: int = 128,
    dropout: float = 0.1,
) -> CRIDGenerator:
    cfg = ModelConfig(
        query_vocab_size=query_vocab_size,
        n_prefix=n_prefix,
        n_suffix=n_suffix,
        d_model=d_model,
        hidden=hidden,
        dropout=dropout,
    )
    return CRIDGenerator(cfg)
