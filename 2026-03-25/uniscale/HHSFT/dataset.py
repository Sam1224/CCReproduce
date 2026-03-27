from __future__ import annotations

import random
from typing import Dict, List

import torch
from torch.utils.data import Dataset


class ToyRankingDataset(Dataset):
    def __init__(
        self,
        num_samples: int = 512,
        num_sparse_fields: int = 8,
        sparse_vocab: int = 1000,
        dense_dim: int = 16,
        history_len_range: tuple[int, int] = (5, 30),
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.num_sparse_fields = num_sparse_fields
        self.sparse_vocab = sparse_vocab
        self.dense_dim = dense_dim
        self.history_len_range = history_len_range
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return self.num_samples

    def _item(self) -> Dict[str, torch.Tensor]:
        sparse = torch.randint(0, self.sparse_vocab, (self.num_sparse_fields,), dtype=torch.long)
        dense = torch.randn(self.dense_dim)
        return {"sparse": sparse, "dense": dense}

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        hist_len = self.rng.randint(*self.history_len_range)

        history_sparse = torch.randint(0, self.sparse_vocab, (hist_len, self.num_sparse_fields), dtype=torch.long)
        history_dense = torch.randn(hist_len, self.dense_dim)

        pos_item = self._item()
        neg_item = self._item()

        # Toy label correlated with cosine similarity between mean history dense and item dense.
        hist_mean = history_dense.mean(dim=0)
        sim_pos = torch.cosine_similarity(hist_mean, pos_item["dense"], dim=0)
        sim_neg = torch.cosine_similarity(hist_mean, neg_item["dense"], dim=0)

        label = torch.tensor(sim_pos > sim_neg, dtype=torch.float32)

        return {
            "history_sparse": history_sparse,
            "history_dense": history_dense,
            "pos_sparse": pos_item["sparse"],
            "pos_dense": pos_item["dense"],
            "neg_sparse": neg_item["sparse"],
            "neg_dense": neg_item["dense"],
            "label": label,
        }


def collate_ranking(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    bsz = len(batch)
    hist_lens = torch.tensor([b["history_sparse"].shape[0] for b in batch], dtype=torch.long)
    max_hist = int(hist_lens.max().item())

    num_fields = batch[0]["history_sparse"].shape[1]
    dense_dim = batch[0]["history_dense"].shape[1]

    history_sparse = torch.zeros(bsz, max_hist, num_fields, dtype=torch.long)
    history_dense = torch.zeros(bsz, max_hist, dense_dim)
    history_mask = torch.zeros(bsz, max_hist, dtype=torch.bool)

    for i, b in enumerate(batch):
        l = b["history_sparse"].shape[0]
        history_sparse[i, :l] = b["history_sparse"]
        history_dense[i, :l] = b["history_dense"]
        history_mask[i, :l] = True

    pos_sparse = torch.stack([b["pos_sparse"] for b in batch], dim=0)
    pos_dense = torch.stack([b["pos_dense"] for b in batch], dim=0)
    neg_sparse = torch.stack([b["neg_sparse"] for b in batch], dim=0)
    neg_dense = torch.stack([b["neg_dense"] for b in batch], dim=0)
    label = torch.stack([b["label"] for b in batch], dim=0)

    return {
        "history_sparse": history_sparse,
        "history_dense": history_dense,
        "history_mask": history_mask,
        "pos_sparse": pos_sparse,
        "pos_dense": pos_dense,
        "neg_sparse": neg_sparse,
        "neg_dense": neg_dense,
        "label": label,
    }
