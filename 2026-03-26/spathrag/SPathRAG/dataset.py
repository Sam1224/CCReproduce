from __future__ import annotations

import random
from typing import Dict, List

import torch
from torch.utils.data import Dataset


class KGQAToyDataset(Dataset):
    def __init__(
        self,
        num_samples: int = 256,
        vocab_size: int = 4096,
        num_paths: int = 6,
        query_len_range: tuple[int, int] = (8, 24),
        path_len_range: tuple[int, int] = (4, 12),
        answer_len_range: tuple[int, int] = (6, 16),
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.num_paths = num_paths
        self.query_len_range = query_len_range
        self.path_len_range = path_len_range
        self.answer_len_range = answer_len_range
        self.rng = random.Random(seed)

        self.bos_id = 1
        self.eos_id = 2

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        lq = self.rng.randint(*self.query_len_range)
        lp = self.rng.randint(*self.path_len_range)
        la = self.rng.randint(*self.answer_len_range)

        query_ids = torch.randint(3, self.vocab_size, (lq,), dtype=torch.long)
        path_ids = torch.randint(3, self.vocab_size, (self.num_paths, lp), dtype=torch.long)

        # Ensure at least one relevant path.
        labels = torch.zeros(self.num_paths, dtype=torch.float32)
        labels[self.rng.randrange(self.num_paths)] = 1.0

        answer = torch.randint(3, self.vocab_size, (la,), dtype=torch.long)
        answer_in = torch.cat([torch.tensor([self.bos_id]), answer[:-1]], dim=0)
        answer_tgt = answer

        return {
            "query_ids": query_ids,
            "path_ids": path_ids,
            "path_labels": labels,
            "answer_in": answer_in,
            "answer_tgt": answer_tgt,
        }


def collate_kgqa(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    bsz = len(batch)

    q_lens = torch.tensor([b["query_ids"].numel() for b in batch], dtype=torch.long)
    a_lens = torch.tensor([b["answer_in"].numel() for b in batch], dtype=torch.long)

    max_q = int(q_lens.max().item())
    max_a = int(a_lens.max().item())

    num_paths = batch[0]["path_ids"].shape[0]
    path_len = batch[0]["path_ids"].shape[1]

    query_ids = torch.zeros(bsz, max_q, dtype=torch.long)
    query_mask = torch.zeros(bsz, max_q, dtype=torch.bool)

    answer_in = torch.zeros(bsz, max_a, dtype=torch.long)
    answer_tgt = torch.full((bsz, max_a), fill_value=-100, dtype=torch.long)
    answer_mask = torch.zeros(bsz, max_a, dtype=torch.bool)

    path_ids = torch.zeros(bsz, num_paths, path_len, dtype=torch.long)
    path_mask = torch.ones(bsz, num_paths, path_len, dtype=torch.bool)

    path_labels = torch.stack([b["path_labels"] for b in batch], dim=0)

    for i, b in enumerate(batch):
        lq = b["query_ids"].numel()
        query_ids[i, :lq] = b["query_ids"]
        query_mask[i, :lq] = True

        la = b["answer_in"].numel()
        answer_in[i, :la] = b["answer_in"]
        answer_tgt[i, :la] = b["answer_tgt"]
        answer_mask[i, :la] = True

        path_ids[i] = b["path_ids"]

    return {
        "query_ids": query_ids,
        "query_mask": query_mask,
        "path_ids": path_ids,
        "path_mask": path_mask,
        "path_labels": path_labels,
        "answer_in": answer_in,
        "answer_tgt": answer_tgt,
        "answer_mask": answer_mask,
    }
