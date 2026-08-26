from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


@dataclass
class SyntheticWorld:
    item_cf: torch.Tensor  # [num_items, d]
    semantic_raw: torch.Tensor  # [num_items, sem_dim]
    semantic_smooth: torch.Tensor  # [num_items, sem_dim]
    item_cluster: torch.Tensor  # [num_items]
    popularity: torch.Tensor  # [num_items]
    user_cf: torch.Tensor  # [num_users, d]
    user_sem: torch.Tensor  # [num_users, sem_dim]
    user_gate: torch.Tensor  # [num_users]


def spectral_smooth(x: torch.Tensor, item_cf: torch.Tensor, topk: int = 12, alpha: float = 0.72, steps: int = 3) -> torch.Tensor:
    """Smooth semantic features on a collaborative item graph.

    This is a small proxy for spectral low-pass filtering: build a kNN graph from
    collaborative item factors and repeatedly apply normalized adjacency.
    """

    sim = F.normalize(item_cf, dim=-1) @ F.normalize(item_cf, dim=-1).t()
    idx = torch.topk(sim, k=min(topk + 1, sim.shape[0]), dim=-1).indices
    adj = torch.zeros_like(sim)
    adj.scatter_(1, idx, 1.0)
    adj = ((adj + adj.t()) > 0).float()
    adj.fill_diagonal_(1.0)

    deg = adj.sum(dim=1).clamp_min(1.0)
    norm_adj = adj / torch.sqrt(deg[:, None] * deg[None, :])
    z = x
    for _ in range(steps):
        z = (1.0 - alpha) * x + alpha * (norm_adj @ z)
    return F.normalize(z, dim=-1)


def build_world(seed: int = 26, num_users: int = 700, num_items: int = 520, d: int = 32, sem_dim: int = 48, clusters: int = 10) -> SyntheticWorld:
    g = torch.Generator().manual_seed(seed)

    cluster_cf = F.normalize(torch.randn(clusters, d, generator=g), dim=-1)
    cluster_sem = F.normalize(torch.randn(clusters, sem_dim, generator=g), dim=-1)

    item_cluster = torch.randint(0, clusters, (num_items,), generator=g)
    item_cf = cluster_cf[item_cluster] + 0.20 * torch.randn(num_items, d, generator=g)
    item_cf = F.normalize(item_cf, dim=-1)

    # LLM-like semantics: useful coarse signal plus spurious high-frequency noise.
    semantic_raw = cluster_sem[item_cluster] + 0.55 * torch.randn(num_items, sem_dim, generator=g)
    semantic_raw = F.normalize(semantic_raw, dim=-1)
    semantic_smooth = spectral_smooth(semantic_raw, item_cf)

    popularity = torch.sigmoid(1.2 * torch.randn(num_items, generator=g))

    preferred_cluster = torch.randint(0, clusters, (num_users,), generator=g)
    user_cf = cluster_cf[preferred_cluster] + 0.25 * torch.randn(num_users, d, generator=g)
    user_sem = cluster_sem[preferred_cluster] + 0.25 * torch.randn(num_users, sem_dim, generator=g)
    user_cf = F.normalize(user_cf, dim=-1)
    user_sem = F.normalize(user_sem, dim=-1)
    user_gate = torch.randn(num_users, generator=g) * 0.7  # positive => more collaborative

    return SyntheticWorld(item_cf, semantic_raw, semantic_smooth, item_cluster, popularity, user_cf, user_sem, user_gate)


def _oracle_scores(world: SyntheticWorld, user_id: int, cand: torch.Tensor) -> torch.Tensor:
    cf = world.item_cf[cand] @ world.user_cf[user_id]
    sem = world.semantic_smooth[cand] @ world.user_sem[user_id]
    gate = torch.sigmoid(world.user_gate[user_id])
    return gate * cf + (1.0 - gate) * sem + 0.05 * world.popularity[cand]


def _sample_candidates(world: SyntheticWorld, user_id: int, k: int, g: torch.Generator) -> torch.Tensor:
    num_items = world.item_cf.shape[0]
    user_cluster = int(torch.argmax(world.user_cf[user_id] @ F.normalize(torch.stack([world.item_cf[world.item_cluster == c].mean(0) for c in range(int(world.item_cluster.max()) + 1)]), dim=-1).t()).item())
    hard_pool = torch.where(world.item_cluster == user_cluster)[0]
    hard = hard_pool[torch.randint(0, len(hard_pool), (k // 2,), generator=g)] if len(hard_pool) else torch.empty(0, dtype=torch.long)
    rand = torch.randint(0, num_items, (k - hard.numel(),), generator=g)
    return torch.cat([hard, rand])[:k]


class UniSpecDataset(Dataset):
    def __init__(self, examples: List[Tuple[int, torch.Tensor, int]]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        user_id, cand, label = self.examples[idx]
        return {"user_id": torch.tensor(user_id), "cand_item_ids": cand.long(), "label": torch.tensor(label)}


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "user_id": torch.stack([b["user_id"] for b in batch]).long(),
        "cand_item_ids": torch.stack([b["cand_item_ids"] for b in batch]).long(),
        "label": torch.stack([b["label"] for b in batch]).long(),
    }


def build_dataloaders(seed: int = 26, batch_size: int = 128, candidate_size: int = 40, train_examples: int = 10000, val_examples: int = 1600, test_examples: int = 1600) -> Tuple[SyntheticWorld, DataLoader, DataLoader, DataLoader]:
    world = build_world(seed=seed)
    g = torch.Generator().manual_seed(seed + 1)

    def make_examples(n: int, noise: float) -> List[Tuple[int, torch.Tensor, int]]:
        examples: List[Tuple[int, torch.Tensor, int]] = []
        for _ in range(n):
            user_id = int(torch.randint(0, world.user_cf.shape[0], (1,), generator=g).item())
            cand = _sample_candidates(world, user_id, candidate_size, g)
            scores = _oracle_scores(world, user_id, cand)
            if noise > 0:
                scores = scores + noise * torch.randn(scores.shape, generator=g)
            label = int(torch.argmax(scores).item())
            examples.append((user_id, cand, label))
        return examples

    train_ds = UniSpecDataset(make_examples(train_examples, noise=0.08))
    val_ds = UniSpecDataset(make_examples(val_examples, noise=0.0))
    test_ds = UniSpecDataset(make_examples(test_examples, noise=0.0))
    return (
        world,
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn),
    )


def hr_ndcg(scores: torch.Tensor, labels: torch.Tensor, ks: Tuple[int, ...] = (1, 5, 10)) -> Dict[str, float]:
    topk = torch.topk(scores, k=max(ks), dim=-1).indices
    metrics: Dict[str, float] = {}
    for k in ks:
        hit = (topk[:, :k] == labels[:, None]).any(dim=-1).float()
        metrics[f"hr@{k}"] = hit.mean().item()
        ndcg = torch.zeros_like(hit)
        where = torch.where(topk[:, :k] == labels[:, None])
        if where[0].numel():
            ndcg[where[0]] = 1.0 / torch.log2(where[1].float() + 2.0)
        metrics[f"ndcg@{k}"] = ndcg.mean().item()
    return metrics
