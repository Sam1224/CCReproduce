from __future__ import annotations

import pickle
import time
from dataclasses import dataclass
from typing import DefaultDict, Dict, List, Tuple

import torch


@dataclass
class Posting:
    doc_id: int
    tok_id: int
    value: float


class InvertedIndex:
    """Neuron-level inverted index for top-k sparse token vectors.

    postings[u] = list of Posting(doc_id, tok_id, value_d_u)

    This is a pedagogical implementation for toy-scale data.
    """

    def __init__(self):
        self.postings: Dict[int, List[Posting]] = {}
        self.doc_token_count: Dict[int, int] = {}

    def add_doc(self, doc_id: int, doc_idx: torch.Tensor, doc_val: torch.Tensor):
        """doc_idx/doc_val: [Ld, k]"""
        Ld = doc_idx.shape[0]
        self.doc_token_count[doc_id] = int(Ld)

        for tok_id in range(Ld):
            idx_row = doc_idx[tok_id].tolist()
            val_row = doc_val[tok_id].tolist()
            for u, vd in zip(idx_row, val_row):
                if u not in self.postings:
                    self.postings[u] = []
                self.postings[u].append(Posting(doc_id=doc_id, tok_id=tok_id, value=float(vd)))

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"postings": self.postings, "doc_token_count": self.doc_token_count}, f)

    @staticmethod
    def load(path: str) -> "InvertedIndex":
        with open(path, "rb") as f:
            raw = pickle.load(f)
        idx = InvertedIndex()
        idx.postings = raw["postings"]
        idx.doc_token_count = raw["doc_token_count"]
        return idx


def _token_maxsim_by_postings(
    q_idx: torch.Tensor,  # [k]
    q_val: torch.Tensor,  # [k]
    postings: Dict[int, List[Posting]],
) -> Dict[int, float]:
    """Return doc_id -> max_{tok} dot(z_q, z_d_tok) for ONE query token."""

    # (doc_id, tok_id) -> partial dot
    partial: Dict[Tuple[int, int], float] = {}

    for u, vq in zip(q_idx.tolist(), q_val.tolist()):
        for p in postings.get(int(u), []):
            key = (p.doc_id, p.tok_id)
            partial[key] = partial.get(key, 0.0) + float(vq) * float(p.value)

    max_per_doc: Dict[int, float] = {}
    for (doc_id, _tok_id), dot in partial.items():
        cur = max_per_doc.get(doc_id)
        if cur is None or dot > cur:
            max_per_doc[doc_id] = dot

    return max_per_doc


def maxsim_query_by_inverted_index(
    q_idx: torch.Tensor,  # [Lq, k]
    q_val: torch.Tensor,  # [Lq, k]
    q_mask: torch.Tensor,  # [Lq]
    index: InvertedIndex,
) -> Dict[int, float]:
    """Return doc_id -> sum_t MaxSim_t (late interaction)"""

    total: Dict[int, float] = {}
    for t in range(q_idx.shape[0]):
        if not bool(q_mask[t].item()):
            continue
        per_doc = _token_maxsim_by_postings(q_idx[t], q_val[t], index.postings)
        for doc_id, best in per_doc.items():
            total[doc_id] = total.get(doc_id, 0.0) + best

    return total


def build_index(doc_sparse: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[InvertedIndex, float]:
    """doc_sparse: list of (doc_idx[Ld,k], doc_val[Ld,k])"""

    t0 = time.time()
    idx = InvertedIndex()
    for doc_id, (d_idx, d_val) in enumerate(doc_sparse):
        idx.add_doc(doc_id, d_idx, d_val)
    return idx, time.time() - t0
