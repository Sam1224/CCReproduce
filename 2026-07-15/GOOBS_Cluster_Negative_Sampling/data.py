import random
from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import Dataset


@dataclass
class Catalog:
    item_cluster: torch.Tensor
    item_features: torch.Tensor


class ToyRetrievalDataset(Dataset):
    def __init__(self, num_users: int = 600, num_items: int = 1200, num_clusters: int = 24, n: int = 6000, seed: int = 5):
        g = torch.Generator().manual_seed(seed)
        self.num_users = num_users
        self.num_items = num_items
        self.num_clusters = num_clusters
        self.n = n
        self.item_cluster = torch.arange(num_items) % num_clusters
        self.item_features = torch.randn(num_items, 8, generator=g)
        self.user_pref = torch.randint(0, num_clusters, (num_users,), generator=g)
        rng = random.Random(seed)
        self.samples = []
        for _ in range(n):
            user = rng.randrange(num_users)
            cluster = int(self.user_pref[user]) if rng.random() < 0.85 else rng.randrange(num_clusters)
            candidates = (self.item_cluster == cluster).nonzero(as_tuple=False).view(-1)
            item = int(candidates[rng.randrange(len(candidates))])
            self.samples.append((user, item, cluster))

    @property
    def catalog(self) -> Catalog:
        return Catalog(self.item_cluster.clone(), self.item_features.clone())

    def __len__(self):
        return self.n

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        u, i, c = self.samples[idx]
        return {"user_id": torch.tensor(u), "item_id": torch.tensor(i), "cluster_id": torch.tensor(c)}


class OnlineItemPool:
    def __init__(self, item_cluster: torch.Tensor, max_per_cluster: int = 128):
        self.max_per_cluster = max_per_cluster
        self.pool: Dict[int, List[int]] = {}
        for item_id, cluster_id in enumerate(item_cluster.tolist()):
            self.pool.setdefault(cluster_id, []).append(item_id)
        for cluster_id in self.pool:
            self.pool[cluster_id] = self.pool[cluster_id][-max_per_cluster:]

    def update(self, item_ids: torch.Tensor, cluster_ids: torch.Tensor) -> None:
        for item_id, cluster_id in zip(item_ids.tolist(), cluster_ids.tolist()):
            bucket = self.pool.setdefault(int(cluster_id), [])
            bucket.append(int(item_id))
            if len(bucket) > self.max_per_cluster:
                del bucket[0 : len(bucket) - self.max_per_cluster]

    def sample_cluster_negatives(self, positives: torch.Tensor, clusters: torch.Tensor, num_neg: int, device) -> torch.Tensor:
        rows = []
        for pos, cluster in zip(positives.tolist(), clusters.tolist()):
            bucket = [x for x in self.pool.get(int(cluster), []) if x != int(pos)]
            if not bucket:
                bucket = [x for values in self.pool.values() for x in values if x != int(pos)]
            rows.append([random.choice(bucket) for _ in range(num_neg)])
        return torch.tensor(rows, dtype=torch.long, device=device)


def make_loaders(batch_size: int = 128):
    ds = ToyRetrievalDataset()
    train, test = torch.utils.data.random_split(ds, [5000, 1000], generator=torch.Generator().manual_seed(13))
    return (
        torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True),
        torch.utils.data.DataLoader(test, batch_size=batch_size),
        ds.catalog,
        ds.num_users,
        ds.num_items,
    )
