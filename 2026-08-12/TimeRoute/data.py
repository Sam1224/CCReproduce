from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class MultiModalCatalog:
    text_embeddings: torch.Tensor
    image_embeddings: torch.Tensor
    audio_embeddings: torch.Tensor
    graph_edges: torch.Tensor


class TimeRouteDataset(Dataset):
    def __init__(self, sessions: torch.Tensor, timestamps: torch.Tensor, targets: torch.Tensor):
        self.sessions = sessions
        self.timestamps = timestamps
        self.targets = targets

    def __len__(self) -> int:
        return self.sessions.size(0)

    def __getitem__(self, index: int):
        return {
            "session": self.sessions[index],
            "timestamps": self.timestamps[index],
            "target": self.targets[index],
        }


def build_synthetic_corpus(
    num_sessions: int = 2048,
    num_items: int = 96,
    session_length: int = 10,
    embedding_dim: int = 32,
    seed: int = 7,
) -> Tuple[MultiModalCatalog, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    base = torch.randn(num_items, embedding_dim, generator=generator)
    text_embeddings = base + 0.15 * torch.randn(num_items, embedding_dim, generator=generator)
    image_embeddings = 0.7 * base + 0.5 * torch.randn(num_items, embedding_dim, generator=generator)
    audio_embeddings = 0.5 * base + 0.7 * torch.randn(num_items, embedding_dim, generator=generator)
    graph_edges = torch.matmul(base, base.t()).softmax(dim=-1)

    sessions = torch.randint(0, num_items, (num_sessions, session_length), generator=generator)
    timestamps = torch.sort(torch.rand(num_sessions, session_length, generator=generator), dim=1).values
    recency = 1.0 - timestamps

    dynamic_modal_weights = torch.stack(
        [
            0.25 + 0.55 * recency.mean(dim=1),
            0.30 + 0.45 * timestamps.mean(dim=1),
            0.20 + 0.35 * recency[:, -3:].mean(dim=1),
        ],
        dim=-1,
    )
    dynamic_modal_weights = dynamic_modal_weights / dynamic_modal_weights.sum(dim=-1, keepdim=True)

    session_text = text_embeddings[sessions].mean(dim=1)
    session_image = image_embeddings[sessions].mean(dim=1)
    session_audio = audio_embeddings[sessions].mean(dim=1)
    fused_session = (
        dynamic_modal_weights[:, 0:1] * session_text
        + dynamic_modal_weights[:, 1:2] * session_image
        + dynamic_modal_weights[:, 2:3] * session_audio
    )
    graph_context = torch.matmul(graph_edges[sessions].mean(dim=1), base)
    scores = torch.matmul(fused_session + 0.2 * graph_context, base.t())
    targets = scores.argmax(dim=-1)

    catalog = MultiModalCatalog(
        text_embeddings=text_embeddings,
        image_embeddings=image_embeddings,
        audio_embeddings=audio_embeddings,
        graph_edges=graph_edges,
    )
    return catalog, sessions, timestamps, targets


def create_dataloaders(batch_size: int = 64, seed: int = 7):
    catalog, sessions, timestamps, targets = build_synthetic_corpus(seed=seed)
    split = int(0.8 * sessions.size(0))
    train_dataset = TimeRouteDataset(sessions[:split], timestamps[:split], targets[:split])
    test_dataset = TimeRouteDataset(sessions[split:], timestamps[split:], targets[split:])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return catalog, train_loader, test_loader
