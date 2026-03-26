from __future__ import annotations

import random
from typing import Dict, List

import torch
from torch.utils.data import Dataset


class OneSearchV2ToyDataset(Dataset):
    def __init__(
        self,
        num_samples: int = 256,
        vocab_size: int = 5000,
        cot_len: int = 8,
        query_len_range: tuple[int, int] = (8, 24),
        context_len_range: tuple[int, int] = (16, 64),
        answer_len_range: tuple[int, int] = (12, 32),
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.cot_len = cot_len
        self.query_len_range = query_len_range
        self.context_len_range = context_len_range
        self.answer_len_range = answer_len_range
        self.rng = random.Random(seed)

        self.bos_id = 1

    def __len__(self) -> int:
        return self.num_samples

    def _make_seq(self, length: int) -> torch.Tensor:
        return torch.randint(3, self.vocab_size, (length,), dtype=torch.long)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        lq = self.rng.randint(*self.query_len_range)
        lc = self.rng.randint(*self.context_len_range)
        la = self.rng.randint(*self.answer_len_range)

        query_ids = self._make_seq(lq)
        context_ids = self._make_seq(lc)

        answer = self._make_seq(la)
        answer_in = torch.cat([torch.tensor([self.bos_id]), answer[:-1]], dim=0)
        answer_tgt = answer

        cot_tgt = self._make_seq(self.cot_len)

        # Toy reward correlated with overlap between query and answer tokens.
        overlap = len(set(query_ids.tolist()) & set(answer_tgt.tolist()))
        reward = torch.tensor([float(overlap) / max(la, 1)], dtype=torch.float32)

        return {
            "query_ids": query_ids,
            "context_ids": context_ids,
            "answer_in": answer_in,
            "answer_tgt": answer_tgt,
            "cot_tgt": cot_tgt,
            "reward": reward,
        }


def collate_onesearchv2(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    bsz = len(batch)

    q_lens = torch.tensor([b["query_ids"].numel() for b in batch], dtype=torch.long)
    c_lens = torch.tensor([b["context_ids"].numel() for b in batch], dtype=torch.long)
    a_lens = torch.tensor([b["answer_in"].numel() for b in batch], dtype=torch.long)

    max_q = int(q_lens.max().item())
    max_c = int(c_lens.max().item())
    max_a = int(a_lens.max().item())

    cot_len = batch[0]["cot_tgt"].numel()

    query_ids = torch.zeros(bsz, max_q, dtype=torch.long)
    query_mask = torch.zeros(bsz, max_q, dtype=torch.bool)

    context_ids = torch.zeros(bsz, max_c, dtype=torch.long)
    context_mask = torch.zeros(bsz, max_c, dtype=torch.bool)

    answer_in = torch.zeros(bsz, max_a, dtype=torch.long)
    answer_tgt = torch.full((bsz, max_a), fill_value=-100, dtype=torch.long)
    answer_mask = torch.zeros(bsz, max_a, dtype=torch.bool)

    cot_tgt = torch.stack([b["cot_tgt"] for b in batch], dim=0)
    reward = torch.cat([b["reward"] for b in batch], dim=0)

    for i, b in enumerate(batch):
        lq = b["query_ids"].numel()
        query_ids[i, :lq] = b["query_ids"]
        query_mask[i, :lq] = True

        lc = b["context_ids"].numel()
        context_ids[i, :lc] = b["context_ids"]
        context_mask[i, :lc] = True

        la = b["answer_in"].numel()
        answer_in[i, :la] = b["answer_in"]
        answer_tgt[i, :la] = b["answer_tgt"]
        answer_mask[i, :la] = True

    return {
        "query_ids": query_ids,
        "query_mask": query_mask,
        "context_ids": context_ids,
        "context_mask": context_mask,
        "answer_in": answer_in,
        "answer_tgt": answer_tgt,
        "answer_mask": answer_mask,
        "cot_tgt": cot_tgt,
        "reward": reward,
    }
