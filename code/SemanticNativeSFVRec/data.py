"""Toy dataset for SemanticNativeSFVRec (interface-aligned with paper's data format)."""

import torch
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import List


@dataclass
class SFVSample:
    """One training sample: user video history → next video to watch."""
    history_content_embs: List[List[float]]  # [L, content_dim] content embeddings of history
    target_content_emb: List[float]          # [content_dim] content embedding of target


class SFVDataset(Dataset):
    """
    Toy short-form video dataset.

    In production, content_emb comes from a frozen content encoder (e.g., CLIP/video BERT).
    Here we simulate them with random unit vectors per "video cluster" to model the
    fact that semantically similar videos share prefix SIDs after quantization.
    """

    def __init__(
        self,
        num_users: int = 2000,
        max_history: int = 50,
        content_dim: int = 256,
        num_video_clusters: int = 500,  # clusters of semantically similar videos
        videos_per_cluster: int = 200,
        seed: int = 42,
    ):
        torch.manual_seed(seed)
        self.content_dim = content_dim

        # Simulate content embeddings: cluster centroids + small noise
        cluster_centers = F.normalize(
            torch.randn(num_video_clusters, content_dim), dim=-1
        )

        self.samples = []
        for _ in range(num_users):
            hist_len = torch.randint(5, max_history, (1,)).item()
            # Sample video clusters for history
            hist_clusters = torch.randint(0, num_video_clusters, (hist_len,))
            hist_embs = cluster_centers[hist_clusters] + 0.1 * torch.randn(hist_len, content_dim)
            hist_embs = F.normalize(hist_embs, dim=-1)

            # Target: sample a video from same cluster as last history item (engagement signal)
            tgt_cluster = hist_clusters[-1].item()
            tgt_emb = cluster_centers[tgt_cluster] + 0.1 * torch.randn(content_dim)
            tgt_emb = F.normalize(tgt_emb, dim=0)

            self.samples.append(SFVSample(
                history_content_embs=hist_embs.tolist(),
                target_content_emb=tgt_emb.tolist(),
            ))

        self.cluster_centers = cluster_centers

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


import torch.nn.functional as F


def collate_fn(batch: List[SFVSample], max_history: int = 50):
    """Pad and collate a batch."""
    content_dim = len(batch[0].history_content_embs[0])

    hist_embs = torch.zeros(len(batch), max_history, content_dim)
    attention_mask = torch.zeros(len(batch), max_history)
    tgt_embs = torch.zeros(len(batch), content_dim)

    for i, s in enumerate(batch):
        hist = s.history_content_embs[-max_history:]  # truncate from the right (most recent)
        L = len(hist)
        hist_tensor = torch.tensor(hist, dtype=torch.float)
        # Right-align: most recent items at the end
        hist_embs[i, max_history - L:] = hist_tensor
        attention_mask[i, max_history - L:] = 1.0
        tgt_embs[i] = torch.tensor(s.target_content_emb, dtype=torch.float)

    return {
        "history_content_embs": hist_embs,    # [B, L, content_dim]
        "attention_mask": attention_mask,       # [B, L]
        "target_content_emb": tgt_embs,        # [B, content_dim]
    }


def get_dataloader(batch_size: int = 32, num_workers: int = 0) -> DataLoader:
    dataset = SFVDataset()
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers,
    )
