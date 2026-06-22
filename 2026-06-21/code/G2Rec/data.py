"""
G2Rec - Data utilities (toy dataset, interface-aligned with real data).

Real datasets: Amazon reviews, ML-1M, etc. For toy data we generate synthetic
user-item interactions that mirror the interface expected by model.py.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def generate_toy_dataset(
    n_users: int = 500,
    n_items: int = 1000,
    seq_len: int = 20,
    interaction_density: float = 0.05,
    seed: int = 42,
):
    """
    Generate a synthetic user-item interaction dataset for G2Rec.

    Returns
    -------
    interactions : dict with keys
        'user_seqs'    : (n_users, seq_len) int array of item IDs (padded with 0)
        'targets'      : (n_users,) int array of next-item targets
        'edge_index'   : (2, E) int array: [user_nodes, item_nodes] for graph
        'n_users'      : int
        'n_items'      : int
    """
    rng = np.random.default_rng(seed)

    # Generate user sequences by sampling items without replacement
    user_seqs = np.zeros((n_users, seq_len), dtype=np.int64)
    targets = np.zeros(n_users, dtype=np.int64)

    for u in range(n_users):
        items = rng.choice(n_items, size=seq_len + 1, replace=False) + 1  # 1-indexed
        user_seqs[u] = items[:seq_len]
        targets[u] = items[seq_len]

    # Build co-engagement graph: items that appear in the same user sequence are connected.
    # Edge format: (user_node_idx, item_node_idx) bipartite graph
    # User nodes: 0..n_users-1; Item nodes: n_users..n_users+n_items-1
    src, dst = [], []
    for u in range(n_users):
        for item_id in user_seqs[u]:
            if item_id > 0:
                src.append(u)
                dst.append(n_users + item_id - 1)  # offset to item node space

    edge_index = np.array([src, dst], dtype=np.int64)

    return {
        "user_seqs": user_seqs,
        "targets": targets,
        "edge_index": edge_index,
        "n_users": n_users,
        "n_items": n_items,
    }


class SeqRecDataset(Dataset):
    """Sequential recommendation dataset."""

    def __init__(self, user_seqs: np.ndarray, targets: np.ndarray):
        self.user_seqs = torch.from_numpy(user_seqs).long()
        self.targets = torch.from_numpy(targets).long()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.user_seqs[idx], self.targets[idx]


def get_dataloaders(data: dict, batch_size: int = 64, val_split: float = 0.1):
    """Split into train/val DataLoaders."""
    n = data["n_users"]
    n_val = int(n * val_split)
    indices = np.arange(n)
    np.random.shuffle(indices)

    train_idx = indices[n_val:]
    val_idx = indices[:n_val]

    train_ds = SeqRecDataset(data["user_seqs"][train_idx], data["targets"][train_idx])
    val_ds = SeqRecDataset(data["user_seqs"][val_idx], data["targets"][val_idx])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
