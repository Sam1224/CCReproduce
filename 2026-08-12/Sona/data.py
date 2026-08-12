from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class Catalog:
    item_embeddings: torch.Tensor


class SessionDataset(Dataset):
    def __init__(self, sessions: torch.Tensor, targets: torch.Tensor):
        self.sessions = sessions
        self.targets = targets

    def __len__(self) -> int:
        return self.sessions.size(0)

    def __getitem__(self, index: int):
        return {
            "session": self.sessions[index],
            "target": self.targets[index],
        }


def build_synthetic_corpus(
    num_sessions: int = 2048,
    num_items: int = 96,
    session_length: int = 12,
    embedding_dim: int = 32,
    seed: int = 42,
) -> Tuple[Catalog, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    item_embeddings = torch.randn(num_items, embedding_dim, generator=generator)
    sessions = torch.randint(0, num_items, (num_sessions, session_length), generator=generator)
    session_signal = item_embeddings[sessions].mean(dim=1)
    scores = torch.matmul(session_signal, item_embeddings.t())
    targets = scores.argmax(dim=-1)
    return Catalog(item_embeddings=item_embeddings), sessions, targets


def create_dataloaders(batch_size: int = 64, seed: int = 42):
    catalog, sessions, targets = build_synthetic_corpus(seed=seed)
    split = int(0.8 * sessions.size(0))
    train_dataset = SessionDataset(sessions[:split], targets[:split])
    test_dataset = SessionDataset(sessions[split:], targets[split:])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return catalog, train_loader, test_loader
