from __future__ import annotations

import random
from typing import Dict, List

import torch
from torch.utils.data import Dataset


class SyntheticMixedToyDataset(Dataset):
    def __init__(
        self,
        num_samples: int = 256,
        vocab_size: int = 4000,
        query_len_range: tuple[int, int] = (6, 16),
        qa_len_range: tuple[int, int] = (12, 32),
        doc_len_range: tuple[int, int] = (32, 96),
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.query_len_range = query_len_range
        self.qa_len_range = qa_len_range
        self.doc_len_range = doc_len_range
        self.rng = random.Random(seed)

        self.bos_id = 1

    def __len__(self) -> int:
        return self.num_samples

    def _make_seq(self, length: int) -> torch.Tensor:
        return torch.randint(3, self.vocab_size, (length,), dtype=torch.long)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        lq = self.rng.randint(*self.query_len_range)
        lqa = self.rng.randint(*self.qa_len_range)
        ldoc = self.rng.randint(*self.doc_len_range)

        query_ids = self._make_seq(lq)

        qa = self._make_seq(lqa)
        qa_in = torch.cat([torch.tensor([self.bos_id]), qa[:-1]], dim=0)
        qa_tgt = qa

        doc = self._make_seq(ldoc)
        doc_in = torch.cat([torch.tensor([self.bos_id]), doc[:-1]], dim=0)
        doc_tgt = doc

        return {"query_ids": query_ids, "qa_in": qa_in, "qa_tgt": qa_tgt, "doc_in": doc_in, "doc_tgt": doc_tgt}


def collate_smt(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    bsz = len(batch)

    q_lens = torch.tensor([b["query_ids"].numel() for b in batch], dtype=torch.long)
    qa_lens = torch.tensor([b["qa_in"].numel() for b in batch], dtype=torch.long)
    doc_lens = torch.tensor([b["doc_in"].numel() for b in batch], dtype=torch.long)

    max_q = int(q_lens.max().item())
    max_qa = int(qa_lens.max().item())
    max_doc = int(doc_lens.max().item())

    query_ids = torch.zeros(bsz, max_q, dtype=torch.long)
    query_mask = torch.zeros(bsz, max_q, dtype=torch.bool)

    qa_in = torch.zeros(bsz, max_qa, dtype=torch.long)
    qa_tgt = torch.full((bsz, max_qa), fill_value=-100, dtype=torch.long)
    qa_mask = torch.zeros(bsz, max_qa, dtype=torch.bool)

    doc_in = torch.zeros(bsz, max_doc, dtype=torch.long)
    doc_tgt = torch.full((bsz, max_doc), fill_value=-100, dtype=torch.long)
    doc_mask = torch.zeros(bsz, max_doc, dtype=torch.bool)

    for i, b in enumerate(batch):
        lq = b["query_ids"].numel()
        query_ids[i, :lq] = b["query_ids"]
        query_mask[i, :lq] = True

        lqa = b["qa_in"].numel()
        qa_in[i, :lqa] = b["qa_in"]
        qa_tgt[i, :lqa] = b["qa_tgt"]
        qa_mask[i, :lqa] = True

        ld = b["doc_in"].numel()
        doc_in[i, :ld] = b["doc_in"]
        doc_tgt[i, :ld] = b["doc_tgt"]
        doc_mask[i, :ld] = True

    return {
        "query_ids": query_ids,
        "query_mask": query_mask,
        "qa_in": qa_in,
        "qa_tgt": qa_tgt,
        "qa_mask": qa_mask,
        "doc_in": doc_in,
        "doc_tgt": doc_tgt,
        "doc_mask": doc_mask,
    }
