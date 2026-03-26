from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast


def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)


@dataclass
class OneSearchV2Output:
    answer_logits: torch.Tensor  # (B,La,V)
    cot_logits: torch.Tensor  # (B,T,V)
    reward_pred: torch.Tensor  # (B,)


class OneModel(nn.Module):
    """OneSearch-V2 (toy).

    Faithful to the *method structure* described in the reproduction stub:
    - Query encoder
    - Latent reasoning tokens (CoT head)
    - Generative decoder conditioned on (query/context/reasoning)
    - Reward head used for preference alignment

    Training can add self-distillation via R-Drop/KL between two stochastic passes.
    """

    def __init__(
        self,
        vocab_size: int = 5000,
        d_model: int = 256,
        nhead: int = 8,
        num_enc_layers: int = 3,
        num_dec_layers: int = 3,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        cot_len: int = 8,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.cot_len = cot_len

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
        self.query_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_enc_layers)

        self.reason_tokens = nn.Parameter(torch.randn(cot_len, d_model) * 0.02)
        self.reason_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.cot_head = nn.Linear(d_model, vocab_size)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
            activation="gelu",
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_dec_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

        self.reward_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def freeze_modules(self) -> None:
        return

    def get_optim(self, lr: float = 3e-4, weight_decay: float = 0.01) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

    def _masked_mean(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.unsqueeze(-1).type_as(h)
        denom = m.sum(dim=1).clamp_min(1.0)
        return (h * m).sum(dim=1) / denom

    @autocast()
    def forward(self, batch: Dict[str, torch.Tensor]) -> OneSearchV2Output:
        query_ids = batch["query_ids"]  # (B,Lq)
        query_mask = batch["query_mask"]
        context_ids = batch["context_ids"]  # (B,Lc)
        context_mask = batch["context_mask"]
        answer_in = batch["answer_in"]  # (B,La)

        # Encode query
        q_emb = self.embed(query_ids).transpose(0, 1)  # (Lq,B,d)
        q_h = self.query_encoder(q_emb, src_key_padding_mask=~query_mask)

        # Latent reasoning tokens attend to query hidden states
        r = self.reason_tokens.unsqueeze(1).expand(self.cot_len, query_ids.shape[0], self.d_model)
        r_h, _ = self.reason_attn(r, q_h, q_h, key_padding_mask=~query_mask)
        cot_logits = self.cot_head(r_h).transpose(0, 1)  # (B,T,V)

        # Build memory for decoder: [context embeddings, query hidden, reasoning hidden]
        c_emb = self.embed(context_ids).transpose(0, 1)  # (Lc,B,d)
        memory = torch.cat([c_emb, q_h, r_h], dim=0)  # (Lc+Lq+T,B,d)

        # Decode answer
        a_emb = self.embed(answer_in).transpose(0, 1)  # (La,B,d)
        tgt_mask = _causal_mask(answer_in.shape[1], a_emb.device)
        dec_h = self.decoder(a_emb, memory, tgt_mask=tgt_mask)
        answer_logits = self.lm_head(dec_h).transpose(0, 1)  # (B,La,V)

        reward_pred = self.reward_head(r_h.mean(dim=0)).squeeze(-1)  # (B,)

        return OneSearchV2Output(answer_logits=answer_logits, cot_logits=cot_logits, reward_pred=reward_pred)


def rdrop_kl(p_logits: torch.Tensor, q_logits: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # p_logits/q_logits: (B,L,V); mask: (B,L)
    p = F.log_softmax(p_logits, dim=-1)
    q = F.log_softmax(q_logits, dim=-1)
    p_prob = p.exp()
    q_prob = q.exp()

    kl_pq = F.kl_div(p, q_prob, reduction="none").sum(dim=-1)
    kl_qp = F.kl_div(q, p_prob, reduction="none").sum(dim=-1)

    kl = (kl_pq + kl_qp) * 0.5
    return kl[mask].mean()


def onesearch_v2_loss(
    out: OneSearchV2Output,
    answer_tgt: torch.Tensor,
    answer_mask: torch.Tensor,
    cot_tgt: torch.Tensor,
    reward: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    lm = F.cross_entropy(out.answer_logits.reshape(-1, out.answer_logits.shape[-1]), answer_tgt.reshape(-1), ignore_index=-100)
    cot = F.cross_entropy(out.cot_logits.reshape(-1, out.cot_logits.shape[-1]), cot_tgt.reshape(-1))
    rew = F.mse_loss(out.reward_pred, reward)

    total = lm + cot + rew
    return total, {"lm_loss": float(lm.detach()), "cot_loss": float(cot.detach()), "reward_loss": float(rew.detach())}
