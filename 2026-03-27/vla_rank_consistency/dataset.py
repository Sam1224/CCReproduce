
import torch
from torch.utils.data import Dataset
import random

class ToyDataset(Dataset):
    """Toy dataset for Beyond Attention Magnitude: Leveraging Inter-layer Rank Consistency for Efficient Vision-Language-Action Models"""
    def __init__(self, num_samples=100):
        self.num_samples = num_samples
    def __len__(self):
        return self.num_samples
    def __getitem__(self, idx):
        return {
            "input_ids": torch.randint(0, 1000, (32,)),
            "labels": torch.randint(0, 1000, (32,)),
            "visual_feats": torch.randn(10, 512) if "Vision" in "Beyond Attention Magnitude: Leveraging Inter-layer Rank Consistency for Efficient Vision-Language-Action Models" else None
        }

def collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "visual_feats": torch.stack([b["visual_feats"] for b in batch]) if batch[0]["visual_feats"] is not None else None
    }
