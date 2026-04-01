from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ModelConfig:
    num_items: int
    embed_dim: int = 64
    hidden_dim: int = 64
    num_layers: int = 1
    dropout: float = 0.1
    use_sparse_moe: bool = False
    moe_num_experts: int = 4
    moe_top_k: int = 1


class SparseMoE(nn.Module):
    def __init__(self, dim: int, num_experts: int = 4, top_k: int = 1):
        super().__init__()
        if top_k != 1:
            # Paper supports sparse MoE; this toy reproduction implements top-1 routing.
            raise ValueError("This toy SparseMoE only implements top_k=1")

        self.top_k = top_k
        self.gate = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim)) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.gate(x)
        expert_id = torch.argmax(logits, dim=-1)
        out = torch.empty_like(x)
        for idx, expert in enumerate(self.experts):
            mask = expert_id == idx
            if mask.any():
                out[mask] = expert(x[mask])
        return out


class SeqRecModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.item_embed = nn.Embedding(cfg.num_items, cfg.embed_dim)
        self.item_mm_embed = nn.Embedding(cfg.num_items, cfg.embed_dim)

        self.gru = nn.GRU(
            input_size=cfg.embed_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        self.out_proj = nn.Linear(cfg.hidden_dim, cfg.embed_dim)

        self.sparse_moe = SparseMoE(cfg.embed_dim, cfg.moe_num_experts, cfg.moe_top_k) if cfg.use_sparse_moe else None

    def encode_sequence(self, seq_item_ids: torch.Tensor) -> torch.Tensor:
        base = self.item_embed(seq_item_ids)
        mm = self.item_mm_embed(seq_item_ids)
        x = base + mm
        if self.sparse_moe is not None:
            x = self.sparse_moe(x)
        h, _ = self.gru(x)
        user_state = h[:, -1, :]
        return self.out_proj(user_state)

    def item_vectors(self) -> torch.Tensor:
        base = self.item_embed.weight
        mm = self.item_mm_embed.weight
        v = base + mm
        if self.sparse_moe is not None:
            v = self.sparse_moe(v)
        return v

    def score_all_items(self, seq_item_ids: torch.Tensor) -> torch.Tensor:
        u = self.encode_sequence(seq_item_ids)
        v = self.item_vectors()
        return u @ v.t()

    def score_items(self, seq_item_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        u = self.encode_sequence(seq_item_ids)
        v = self.item_vectors()[item_ids]
        return (u * v).sum(dim=-1)
