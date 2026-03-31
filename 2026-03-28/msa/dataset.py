from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Tuple

import numpy as np
import torch


@dataclass
class Sample:
    docs: np.ndarray  # [N_docs, L]
    query: np.ndarray  # [Q]
    answer: int


class MemoryQADataset:
    def __init__(
        self,
        *,
        n_samples: int,
        n_docs: int = 32,
        doc_len: int = 32,
        vocab_size: int = 256,
        seed: int = 0,
    ):
        self.n_docs = n_docs
        self.doc_len = doc_len
        self.vocab_size = vocab_size
        self._rng = np.random.default_rng(seed)
        self._samples = [self._make_one() for _ in range(n_samples)]

    def _make_one(self) -> Sample:
        rng = self._rng

        # Token 0 reserved for PAD.
        key = int(rng.integers(1, self.vocab_size // 2))
        value = int(rng.integers(self.vocab_size // 2, self.vocab_size))

        docs = rng.integers(1, self.vocab_size, size=(self.n_docs, self.doc_len), dtype=np.int64)

        target_doc = int(rng.integers(0, self.n_docs))
        pos = int(rng.integers(0, self.doc_len - 2))
        docs[target_doc, pos] = key
        docs[target_doc, pos + 1] = value

        # Query: [key, filler, filler]
        query = np.array([key, int(rng.integers(1, self.vocab_size)), int(rng.integers(1, self.vocab_size))], dtype=np.int64)

        return Sample(docs=docs, query=query, answer=value)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Sample:
        return self._samples[idx]


@dataclass
class Batch:
    docs: torch.Tensor  # [B,N_docs,L]
    query: torch.Tensor  # [B,Q]
    answer: torch.Tensor  # [B]


def collate(samples: List[Sample]) -> Batch:
    docs = torch.from_numpy(np.stack([s.docs for s in samples], axis=0)).long()
    query = torch.from_numpy(np.stack([s.query for s in samples], axis=0)).long()
    answer = torch.tensor([s.answer for s in samples]).long()
    return Batch(docs=docs, query=query, answer=answer)


def batch_iter(ds: MemoryQADataset, *, batch_size: int, seed: int = 0) -> Iterator[Batch]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(ds))
    while True:
        rng.shuffle(idx)
        for i in range(0, len(idx), batch_size):
            chunk = idx[i : i + batch_size]
            yield collate([ds[int(j)] for j in chunk])
