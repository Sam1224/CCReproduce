from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast


def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)


class CausalLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
            activation="gelu",
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor, attn_mask: torch.Tensor, extra_bias: torch.Tensor | None = None) -> torch.Tensor:
        # input_ids: (B,L)
        x = self.embed(input_ids)  # (B,L,d)
        if extra_bias is not None:
            x = x + extra_bias.unsqueeze(1)

        x = x.transpose(0, 1)  # (L,B,d)
        key_padding_mask = ~attn_mask
        tgt_mask = _causal_mask(input_ids.shape[1], x.device)

        h = self.blocks(x, mask=tgt_mask, src_key_padding_mask=key_padding_mask)
        logits = self.lm_head(h).transpose(0, 1)  # (B,L,V)
        return logits


@dataclass
class SMTOutput:
    qa_logits: torch.Tensor
    doc_logits: torch.Tensor


class OneModel(nn.Module):
    """Synthetic Mixed Training + Focal Rewriting (toy).

    Key structure:
    - Two synthetic streams: QA and Documents.
    - Focal rewriting: question-conditioned bias injected into document stream.
    """

    def __init__(
        self,
        vocab_size: int = 4000,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.query_embed = nn.Embedding(vocab_size, d_model)

        self.qa_lm = CausalLM(vocab_size, d_model, nhead, num_layers, dim_feedforward, dropout)
        self.doc_lm = CausalLM(vocab_size, d_model, nhead, num_layers, dim_feedforward, dropout)

        self.focal_gate = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model), nn.Sigmoid())

    def freeze_modules(self) -> None:
        return

    def get_optim(self, lr: float = 3e-4, weight_decay: float = 0.01) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

    def _query_summary(self, query_ids: torch.Tensor, query_mask: torch.Tensor) -> torch.Tensor:
        q = self.query_embed(query_ids)  # (B,Lq,d)
        m = query_mask.unsqueeze(-1).type_as(q)
        denom = m.sum(dim=1).clamp_min(1.0)
        return (q * m).sum(dim=1) / denom

    @autocast()
    def forward(self, batch: Dict[str, torch.Tensor]) -> SMTOutput:
        query_ids = batch["query_ids"]
        query_mask = batch["query_mask"]
        qa_in = batch["qa_in"]
        qa_mask = batch["qa_mask"]
        doc_in = batch["doc_in"]
        doc_mask = batch["doc_mask"]

        q_sum = self._query_summary(query_ids, query_mask)  # (B,d)
        focal_bias = self.focal_gate(q_sum) * q_sum  # (B,d)

        qa_logits = self.qa_lm(qa_in, qa_mask)
        doc_logits = self.doc_lm(doc_in, doc_mask, extra_bias=focal_bias)

        return SMTOutput(qa_logits=qa_logits, doc_logits=doc_logits)


def smt_loss(
    out: SMTOutput,
    qa_tgt: torch.Tensor,
    doc_tgt: torch.Tensor,
    qa_weight: float = 1.0,
    doc_weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    qa_ce = F.cross_entropy(out.qa_logits.reshape(-1, out.qa_logits.shape[-1]), qa_tgt.reshape(-1), ignore_index=-100)
    doc_ce = F.cross_entropy(out.doc_logits.reshape(-1, out.doc_logits.shape[-1]), doc_tgt.reshape(-1), ignore_index=-100)
    total = qa_weight * qa_ce + doc_weight * doc_ce
    return total, {"qa_loss": float(qa_ce.detach()), "doc_loss": float(doc_ce.detach())}
