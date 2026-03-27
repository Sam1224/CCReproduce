from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast


@dataclass
class SPathRAGOutput:
    answer_logits: torch.Tensor  # (B,La,V)
    path_scores: torch.Tensor  # (B,K)


def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    # True means masked (disallowed)
    return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)


class OneModel(nn.Module):
    """S-Path-RAG toy pipeline.

    Implements the key computational structure:
    - Encode query
    - Encode K candidate paths
    - Score / mix paths (semantic-aware latent mixture)
    - Condition a generative decoder on (query, mixed_path)
    """

    def __init__(
        self,
        vocab_size: int = 4096,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.embed = nn.Embedding(vocab_size, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
            activation="gelu",
            norm_first=True,
        )
        self.query_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.path_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
            activation="gelu",
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)

        self.path_scorer = nn.Linear(d_model, d_model, bias=False)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def freeze_modules(self) -> None:
        return

    def get_optim(self, lr: float = 3e-4, weight_decay: float = 0.01) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

    def _masked_mean(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # h: (B,L,d) mask: (B,L)
        m = mask.unsqueeze(-1).type_as(h)
        denom = m.sum(dim=1).clamp_min(1.0)
        return (h * m).sum(dim=1) / denom

    @autocast()
    def forward(self, batch: Dict[str, torch.Tensor]) -> SPathRAGOutput:
        query_ids = batch["query_ids"]  # (B,Lq)
        query_mask = batch["query_mask"]  # (B,Lq)
        path_ids = batch["path_ids"]  # (B,K,Lp)
        answer_in = batch["answer_in"]  # (B,La)

        bsz, lq = query_ids.shape
        k = path_ids.shape[1]
        lp = path_ids.shape[2]
        la = answer_in.shape[1]

        # Encode query
        q_emb = self.embed(query_ids).transpose(0, 1)  # (Lq,B,d)
        q_h = self.query_encoder(q_emb, src_key_padding_mask=~query_mask)  # (Lq,B,d)
        q_h_b = q_h.transpose(0, 1)  # (B,Lq,d)
        q_rep = self._masked_mean(q_h_b, query_mask)  # (B,d)

        # Encode paths
        p_flat = path_ids.reshape(bsz * k, lp)
        p_emb = self.embed(p_flat).transpose(0, 1)  # (Lp,BK,d)
        p_h = self.path_encoder(p_emb)  # (Lp,BK,d)
        p_rep = p_h.mean(dim=0).reshape(bsz, k, self.d_model)  # (B,K,d)

        # Semantic-aware path scoring: dot( W q , p )
        q_proj = self.path_scorer(q_rep).unsqueeze(1)  # (B,1,d)
        path_scores = torch.sum(q_proj * p_rep, dim=-1)  # (B,K)

        weights = F.softmax(path_scores, dim=-1).unsqueeze(-1)  # (B,K,1)
        mixed_path = torch.sum(weights * p_rep, dim=1)  # (B,d)

        # Memory tokens: [query_rep, mixed_path]
        memory = torch.stack([q_rep, mixed_path], dim=0)  # (2,B,d)

        # Decode answer
        a_emb = self.embed(answer_in).transpose(0, 1)  # (La,B,d)
        tgt_mask = _causal_mask(la, a_emb.device)
        dec_h = self.decoder(a_emb, memory, tgt_mask=tgt_mask)
        answer_logits = self.lm_head(dec_h).transpose(0, 1)  # (B,La,V)

        return SPathRAGOutput(answer_logits=answer_logits, path_scores=path_scores)


def spathrag_loss(
    out: SPathRAGOutput,
    answer_tgt: torch.Tensor,
    answer_mask: torch.Tensor,
    path_labels: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    ce = F.cross_entropy(out.answer_logits.reshape(-1, out.answer_logits.shape[-1]), answer_tgt.reshape(-1), ignore_index=-100)
    bce = F.binary_cross_entropy_with_logits(out.path_scores, path_labels)
    total = ce + bce
    return total, {"lm_loss": float(ce.detach()), "path_loss": float(bce.detach())}
