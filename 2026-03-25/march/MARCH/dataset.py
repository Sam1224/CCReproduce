from __future__ import annotations

import random
from typing import Dict, List

import torch
from torch.utils.data import Dataset


class MarchToyDataset(Dataset):
    def __init__(
        self,
        num_samples: int = 256,
        vocab_size: int = 6000,
        query_len_range: tuple[int, int] = (8, 20),
        evidence_len_range: tuple[int, int] = (32, 96),
        response_len_range: tuple[int, int] = (16, 48),
        num_claims: int = 4,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.query_len_range = query_len_range
        self.evidence_len_range = evidence_len_range
        self.response_len_range = response_len_range
        self.num_claims = num_claims
        self.rng = random.Random(seed)

        self.bos_id = 1

    def __len__(self) -> int:
        return self.num_samples

    def _seq(self, length: int) -> torch.Tensor:
        return torch.randint(3, self.vocab_size, (length,), dtype=torch.long)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        lq = self.rng.randint(*self.query_len_range)
        le = self.rng.randint(*self.evidence_len_range)
        lr = self.rng.randint(*self.response_len_range)

        query_ids = self._seq(lq)
        evidence_ids = self._seq(le)

        response = self._seq(lr)
        response_in = torch.cat([torch.tensor([self.bos_id]), response[:-1]], dim=0)
        response_tgt = response

        # Claims are represented by token positions in the response.
        claim_pos = torch.tensor(
            [self.rng.randrange(lr) for _ in range(self.num_claims)],
            dtype=torch.long,
        )
        # Toy supportedness: even token id => supported.
        claim_labels = (response_tgt[claim_pos] % 2 == 0).float()

        return {
            "query_ids": query_ids,
            "evidence_ids": evidence_ids,
            "response_in": response_in,
            "response_tgt": response_tgt,
            "claim_pos": claim_pos,
            "claim_labels": claim_labels,
        }


def collate_march(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    bsz = len(batch)

    q_lens = torch.tensor([b["query_ids"].numel() for b in batch], dtype=torch.long)
    e_lens = torch.tensor([b["evidence_ids"].numel() for b in batch], dtype=torch.long)
    r_lens = torch.tensor([b["response_in"].numel() for b in batch], dtype=torch.long)

    max_q = int(q_lens.max().item())
    max_e = int(e_lens.max().item())
    max_r = int(r_lens.max().item())

    num_claims = batch[0]["claim_pos"].numel()

    query_ids = torch.zeros(bsz, max_q, dtype=torch.long)
    query_mask = torch.zeros(bsz, max_q, dtype=torch.bool)

    evidence_ids = torch.zeros(bsz, max_e, dtype=torch.long)
    evidence_mask = torch.zeros(bsz, max_e, dtype=torch.bool)

    response_in = torch.zeros(bsz, max_r, dtype=torch.long)
    response_tgt = torch.full((bsz, max_r), fill_value=-100, dtype=torch.long)
    response_mask = torch.zeros(bsz, max_r, dtype=torch.bool)

    claim_pos = torch.stack([b["claim_pos"] for b in batch], dim=0)
    claim_labels = torch.stack([b["claim_labels"] for b in batch], dim=0)

    for i, b in enumerate(batch):
        lq = b["query_ids"].numel()
        query_ids[i, :lq] = b["query_ids"]
        query_mask[i, :lq] = True

        le = b["evidence_ids"].numel()
        evidence_ids[i, :le] = b["evidence_ids"]
        evidence_mask[i, :le] = True

        lr = b["response_in"].numel()
        response_in[i, :lr] = b["response_in"]
        response_tgt[i, :lr] = b["response_tgt"]
        response_mask[i, :lr] = True

    return {
        "query_ids": query_ids,
        "query_mask": query_mask,
        "evidence_ids": evidence_ids,
        "evidence_mask": evidence_mask,
        "response_in": response_in,
        "response_tgt": response_tgt,
        "response_mask": response_mask,
        "claim_pos": claim_pos,
        "claim_labels": claim_labels,
    }
