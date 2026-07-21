from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class WhaleConfig:
    num_users: int = 128
    num_items: int = 96
    num_contexts: int = 8
    num_categories: int = 12
    hidden: int = 64
    seq_len: int = 20


class WukongBlock(nn.Module):
    def __init__(self, hidden: int, num_fields: int = 3):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.pair_proj = nn.Linear(num_fields * num_fields, hidden)
        self.mlp = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, hidden))

    def forward(self, fields: torch.Tensor) -> torch.Tensor:
        pooled = fields.mean(dim=1)
        pair = torch.bmm(fields, fields.transpose(1, 2)).flatten(start_dim=1)
        pair_repr = self.pair_proj(pair)
        return self.mlp(torch.cat([self.norm(pooled), pair_repr], dim=-1))


class HSTUBlock(nn.Module):
    def __init__(self, hidden: int, num_heads: int = 4):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=num_heads, dim_feedforward=hidden * 2, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)

    def forward(self, seq_embed: torch.Tensor) -> torch.Tensor:
        return self.encoder(seq_embed)


class WhaleFusion(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)
        self.out = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, hidden))

    def forward(self, static_repr: torch.Tensor, seq_repr: torch.Tensor) -> torch.Tensor:
        q = self.query(static_repr).unsqueeze(1)
        k = self.key(seq_repr)
        v = self.value(seq_repr)
        attn = torch.softmax(torch.matmul(q, k.transpose(1, 2)) / (static_repr.shape[-1] ** 0.5), dim=-1)
        seq_ctx = torch.matmul(attn, v).squeeze(1)
        return self.out(torch.cat([static_repr, seq_ctx], dim=-1))


class WhaleToy(nn.Module):
    def __init__(self, cfg: WhaleConfig = WhaleConfig()):
        super().__init__()
        self.cfg = cfg
        self.user = nn.Embedding(cfg.num_users, cfg.hidden)
        self.item = nn.Embedding(cfg.num_items, cfg.hidden)
        self.context = nn.Embedding(cfg.num_contexts, cfg.hidden)
        self.category = nn.Embedding(cfg.num_categories, cfg.hidden)
        self.sequence = nn.Embedding(cfg.num_items, cfg.hidden)
        self.wukong = WukongBlock(cfg.hidden, num_fields=6)
        self.hstu = HSTUBlock(cfg.hidden)
        self.fusion = WhaleFusion(cfg.hidden)
        self.head = nn.Sequential(nn.Linear(cfg.hidden, cfg.hidden), nn.GELU(), nn.Linear(cfg.hidden, 1))

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        fields = torch.stack([
            self.user(batch["user_id"]),
            self.item(batch["item_id"]),
            self.context(batch["context_id"]),
            self.category(batch["user_pref"]),
            self.category(batch["item_category"]),
            self.category(batch["focus_category"]),
        ], dim=1)
        static_repr = self.wukong(fields)
        seq_repr = self.hstu(self.sequence(batch["sequence"]))
        fused = self.fusion(static_repr, seq_repr)
        return {"logits": self.head(fused), "fused": fused}


def loss_fn(outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(outputs["logits"], batch["label"])
