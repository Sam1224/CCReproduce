"""
Toy data generation for FLUID reproduction.

Simulates the data format for industrial livestreaming recommendation:
- Multimodal content features (simulated image/video/text embeddings)
- User interaction sequences
- Room-level and slice-level temporal information
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


def generate_toy_data(
    n_rooms: int = 1000,
    n_users: int = 5000,
    n_interactions: int = 50000,
    feature_dim: int = 256,
    seed: int = 42,
):
    """Generate toy dataset mimicking livestreaming recommendation data."""
    rng = np.random.default_rng(seed)

    # Simulate multimodal content features for each room
    # In the paper, these come from a cross-domain encoder trained on short videos + livestreams
    room_visual_features = rng.standard_normal((n_rooms, feature_dim)).astype(np.float32)
    room_audio_features = rng.standard_normal((n_rooms, feature_dim // 2)).astype(np.float32)
    room_text_features = rng.standard_normal((n_rooms, feature_dim // 2)).astype(np.float32)

    # Simulate user interaction sequences
    user_ids = rng.integers(0, n_users, size=n_interactions)
    room_ids = rng.integers(0, n_rooms, size=n_interactions)
    # Label: click (1) or skip (0)
    labels = rng.integers(0, 2, size=n_interactions)

    return {
        "room_visual": room_visual_features,
        "room_audio": room_audio_features,
        "room_text": room_text_features,
        "user_ids": user_ids,
        "room_ids": room_ids,
        "labels": labels,
        "n_rooms": n_rooms,
        "n_users": n_users,
    }


class LivestreamingDataset(Dataset):
    """Dataset for livestreaming recommendation with multimodal content features."""

    def __init__(self, data: dict, split: str = "train", train_ratio: float = 0.8):
        n = len(data["user_ids"])
        split_idx = int(n * train_ratio)

        if split == "train":
            idx = slice(None, split_idx)
        else:
            idx = slice(split_idx, None)

        self.user_ids = torch.tensor(data["user_ids"][idx], dtype=torch.long)
        self.room_ids = torch.tensor(data["room_ids"][idx], dtype=torch.long)
        self.labels = torch.tensor(data["labels"][idx], dtype=torch.float32)

        # Store room features as tensors
        self.room_visual = torch.tensor(data["room_visual"], dtype=torch.float32)
        self.room_audio = torch.tensor(data["room_audio"], dtype=torch.float32)
        self.room_text = torch.tensor(data["room_text"], dtype=torch.float32)

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        room_id = self.room_ids[idx]
        return {
            "user_id": self.user_ids[idx],
            "room_id": room_id,
            "visual_feat": self.room_visual[room_id],
            "audio_feat": self.room_audio[room_id],
            "text_feat": self.room_text[room_id],
            "label": self.labels[idx],
        }


def get_dataloaders(batch_size: int = 256, **data_kwargs):
    data = generate_toy_data(**data_kwargs)
    train_ds = LivestreamingDataset(data, split="train")
    val_ds = LivestreamingDataset(data, split="val")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, data
