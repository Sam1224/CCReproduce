from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 96):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        mask = (ids != 0).float().unsqueeze(-1)
        x = self.emb(ids)
        x = (x * mask).sum(-2) / mask.sum(-2).clamp_min(1.0)
        return F.normalize(self.proj(x), dim=-1)


class MemoryAbstraction(nn.Module):
    def __init__(self, dim: int = 96):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.update = nn.GRUCell(dim, dim)

    def forward(self, behavior_emb: torch.Tensor) -> torch.Tensor:
        bsz, steps, dim = behavior_emb.shape
        state = torch.zeros(bsz, dim, device=behavior_emb.device)
        for t in range(steps):
            x = behavior_emb[:, t]
            state = self.update(x * self.gate(x), state)
        return F.normalize(state, dim=-1)


class QueryAgentR1(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 96):
        super().__init__()
        self.encoder = TextEncoder(vocab_size, dim)
        self.memory = MemoryAbstraction(dim)
        self.scorer = nn.Sequential(nn.Linear(dim * 3, dim), nn.GELU(), nn.Linear(dim, 1))

    def forward(self, history_ids: torch.Tensor, candidate_ids: torch.Tensor) -> torch.Tensor:
        bsz, h, l = history_ids.shape
        c = candidate_ids.size(1)
        hist_emb = self.encoder(history_ids.view(bsz * h, l)).view(bsz, h, -1)
        mem = self.memory(hist_emb)
        cand_emb = self.encoder(candidate_ids.view(bsz * c, l)).view(bsz, c, -1)
        mem_rep = mem.unsqueeze(1).expand(-1, c, -1)
        feats = torch.cat([mem_rep, cand_emb, mem_rep * cand_emb], dim=-1)
        return self.scorer(feats).squeeze(-1)


def consistency_reward(selected_queries: list[str], target_sets: list[set[int]], products) -> torch.Tensor:
    rewards = []
    for q, targets in zip(selected_queries, target_sets):
        toks = set(q.split())
        retrieved = [p.pid for p in products if p.category in toks or p.intent in toks][:10]
        hit = float(any(pid in targets for pid in retrieved))
        exactish = float(len(toks) >= 3)
        rewards.append(0.5 * hit + 0.5 * exactish)
    return torch.tensor(rewards)
