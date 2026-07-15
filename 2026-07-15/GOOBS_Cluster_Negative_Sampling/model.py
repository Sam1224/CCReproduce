from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TwoTowerConfig:
    num_users: int
    num_items: int
    dim: int = 64


class TwoTower(nn.Module):
    def __init__(self, cfg: TwoTowerConfig):
        super().__init__()
        self.user = nn.Sequential(nn.Embedding(cfg.num_users, cfg.dim), nn.LayerNorm(cfg.dim))
        self.item = nn.Sequential(nn.Embedding(cfg.num_items, cfg.dim), nn.LayerNorm(cfg.dim))

    def encode_user(self, user_id: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.user(user_id), dim=-1)

    def encode_item(self, item_id: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.item(item_id), dim=-1)

    def sampled_logits(self, user_id: torch.Tensor, pos_item: torch.Tensor, neg_items: torch.Tensor) -> torch.Tensor:
        u = self.encode_user(user_id)
        pos = self.encode_item(pos_item).unsqueeze(1)
        neg = self.encode_item(neg_items)
        items = torch.cat([pos, neg], dim=1)
        return torch.einsum("bd,bkd->bk", u, items) * 20.0


def sampled_softmax_loss(logits: torch.Tensor) -> torch.Tensor:
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)


@torch.no_grad()
def hit_rate_at_k(model: TwoTower, loader, num_items: int, k: int, device) -> float:
    model.eval()
    all_items = torch.arange(num_items, device=device)
    item_emb = model.encode_item(all_items)
    hits = 0
    total = 0
    for batch in loader:
        user = batch["user_id"].to(device)
        pos = batch["item_id"].to(device)
        scores = model.encode_user(user) @ item_emb.t()
        topk = scores.topk(k=min(k, num_items), dim=-1).indices
        hits += (topk == pos[:, None]).any(dim=1).sum().item()
        total += pos.numel()
    return hits / max(total, 1)
