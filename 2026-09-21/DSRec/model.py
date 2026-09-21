from __future__ import annotations

import torch
from torch import nn


class TimeModulatedSSM(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.input_proj = nn.Linear(dim, dim)
        self.state_proj = nn.Linear(dim, dim, bias=False)
        self.gap_proj = nn.Linear(1, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        state = torch.zeros(x.size(0), x.size(-1), device=x.device)
        states = []
        for step in range(x.size(1)):
            gap_gate = torch.sigmoid(self.gap_proj(deltas[:, step : step + 1]))
            proposal = torch.tanh(self.input_proj(x[:, step]) + self.state_proj(state))
            state = gap_gate * proposal + (1.0 - gap_gate) * state
            states.append(self.norm(state))
        return torch.stack(states, dim=1)


class DSRec(nn.Module):
    def __init__(self, num_items: int = 80, dim: int = 64):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, dim, padding_idx=0)
        self.long_encoder = nn.GRU(dim, dim, batch_first=True)
        self.short_encoder = TimeModulatedSSM(dim)
        self.long_to_short = nn.Linear(dim, dim)
        self.short_to_long = nn.Linear(dim, dim)
        self.fusion = nn.Sequential(nn.LayerNorm(dim * 2), nn.Linear(dim * 2, dim), nn.GELU())
        self.head = nn.Linear(dim, num_items)

    def forward(self, items: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        emb = self.item_embedding(items)
        long_states, _ = self.long_encoder(emb)
        short_states = self.short_encoder(emb, deltas)
        long_final = long_states[:, -1] + self.short_to_long(short_states[:, -1])
        short_final = short_states[:, -1] + self.long_to_short(long_states[:, -1])
        fused = self.fusion(torch.cat([long_final, short_final], dim=-1))
        return self.head(fused)
