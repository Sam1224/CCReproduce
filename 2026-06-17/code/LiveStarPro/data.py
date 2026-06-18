"""
LiveStarPro — Data utilities
Toy synthetic dataset of streaming video clips with response annotations.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple


def generate_toy_stream_data(
    n_samples: int = 1000,
    seq_len: int = 16,          # frames per clip
    frame_h: int = 32,
    frame_w: int = 32,
    n_response_classes: int = 5,
    seed: int = 42,
) -> Dict:
    rng = np.random.default_rng(seed)

    # Synthetic frames (random pixel values)
    frames = rng.uniform(0, 1, (n_samples, seq_len, 3, frame_h, frame_w)).astype(np.float32)

    # Random respond frame index (when agent should respond)
    respond_at = rng.integers(0, seq_len, size=n_samples)

    # Random response class (what the response is)
    response_class = rng.integers(0, n_response_classes, size=n_samples)

    return {
        "frames": frames,
        "respond_at": respond_at,
        "response_class": response_class,
        "n_samples": n_samples,
        "seq_len": seq_len,
        "n_response_classes": n_response_classes,
    }


class StreamDataset(Dataset):
    def __init__(self, data: Dict, indices: np.ndarray):
        self.frames = torch.FloatTensor(data["frames"][indices])
        self.respond_at = torch.LongTensor(data["respond_at"][indices])
        self.response_class = torch.LongTensor(data["response_class"][indices])

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return {
            "frames": self.frames[idx],
            "respond_at": self.respond_at[idx],
            "response_class": self.response_class[idx],
        }


def build_dataloaders(data: Dict, train_ratio: float = 0.8, batch_size: int = 16):
    n = data["n_samples"]
    split = int(n * train_ratio)
    ids = np.arange(n)

    def make(ids, shuffle):
        return DataLoader(StreamDataset(data, ids), batch_size=batch_size, shuffle=shuffle)

    return make(ids[:split], True), make(ids[split:], False)


if __name__ == "__main__":
    data = generate_toy_stream_data()
    train_loader, val_loader = build_dataloaders(data)
    batch = next(iter(train_loader))
    print("frames shape:", batch["frames"].shape)
    print("respond_at:", batch["respond_at"][:4])
    print("Data OK.")
