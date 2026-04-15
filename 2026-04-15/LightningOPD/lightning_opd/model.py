from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    emb_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 2


class GRULM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.rnn = nn.GRU(
            input_size=cfg.emb_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
        )
        self.lm_head = nn.Linear(cfg.hidden_dim, cfg.vocab_size)

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        x = self.emb(input_ids)  # [B, T, E]
        h, _ = self.rnn(x)
        return self.lm_head(h)  # [B, T, V]


@torch.no_grad()
def greedy_generate(
    model: nn.Module,
    input_ids: torch.LongTensor,
    eos_id: int,
    max_new_tokens: int,
) -> torch.LongTensor:
    model.eval()
    out = input_ids
    for _ in range(max_new_tokens):
        logits = model(out)
        next_logits = logits[:, -1, :]
        next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
        out = torch.cat([out, next_id], dim=1)
        if torch.all(next_id.squeeze(-1) == eos_id):
            break
    return out
