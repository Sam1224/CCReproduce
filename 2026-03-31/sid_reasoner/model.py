from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalTransformerLM(nn.Module):
    def __init__(
        self,
        *,
        vocab: int,
        d: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        dropout: float = 0.1,
        max_len: int = 256,
    ) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d,
                    nhead=n_heads,
                    dim_feedforward=4 * d,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(n_layers)
            ]
        )
        self.ln = nn.LayerNorm(d)
        self.lm = nn.Linear(d, vocab)

    def forward(self, ids: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
        B, L = ids.shape
        pos = torch.arange(L, device=ids.device).unsqueeze(0).expand(B, L)
        x = self.emb(ids) + self.pos(pos)

        pad_mask = ids == pad_id
        causal = torch.triu(torch.ones(L, L, device=ids.device), diagonal=1).bool()

        for blk in self.blocks:
            x = blk(x, src_mask=causal, src_key_padding_mask=pad_mask)

        x = self.ln(x)
        return self.lm(x)

    @torch.no_grad()
    def generate(
        self,
        *,
        prompt: torch.Tensor,
        max_new_tokens: int,
        eos_id: int,
        temperature: float = 0.9,
        top_p: float = 0.92,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # prompt: (B,L)
        ids = prompt
        B = int(ids.shape[0])
        logp_sum = torch.zeros(B, device=ids.device)

        for _ in range(max_new_tokens):
            logits = self.forward(ids)[:, -1, :]
            logits = logits / max(1e-6, temperature)
            probs = torch.softmax(logits, dim=-1)

            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cdf = torch.cumsum(sorted_probs, dim=-1)
            mask = cdf <= top_p
            mask[:, 0] = True
            filtered = sorted_probs * mask
            filtered = filtered / filtered.sum(dim=-1, keepdim=True)

            next_in_sorted = torch.multinomial(filtered, 1).squeeze(1)
            next_id = sorted_idx[torch.arange(B, device=ids.device), next_in_sorted]

            lp = torch.log(probs.gather(1, next_id.unsqueeze(1)).squeeze(1).clamp_min(1e-9))
            logp_sum = logp_sum + lp

            ids = torch.cat([ids, next_id.unsqueeze(1)], dim=1)
            if (next_id == eos_id).all():
                break

        return ids, logp_sum


def lm_loss(logits: torch.Tensor, ids: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
    tgt = ids[:, 1:].contiguous()
    pred = logits[:, :-1, :].contiguous()
    return F.cross_entropy(pred.view(-1, pred.shape[-1]), tgt.view(-1), ignore_index=pad_id)


def token_logprob(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits[:, :-1, :], dim=-1)
    tgt = ids[:, 1:]
    return logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
