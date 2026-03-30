from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch


def tokenize(text: str) -> List[str]:
    return [t for t in re.split(r"[^A-Za-z0-9]+", text.lower()) if t]


def hash_tokens(tokens: List[str], dim: int) -> np.ndarray:
    v = np.zeros((dim,), dtype=np.float32)
    for t in tokens:
        v[hash(t) % dim] += 1.0
    n = np.linalg.norm(v) + 1e-6
    return v / n


@dataclass
class Retriever:
    dim: int = 256

    def __post_init__(self):
        self.doc_ids: List[str] = []
        self.doc_vecs: np.ndarray = np.zeros((0, self.dim), dtype=np.float32)
        self.doc_texts: List[str] = []

    def build(self, docs: List[Tuple[str, str]]):
        self.doc_ids = [d[0] for d in docs]
        self.doc_texts = [d[1] for d in docs]
        vecs = [hash_tokens(tokenize(text), self.dim) for _, text in docs]
        self.doc_vecs = np.stack(vecs, axis=0)

    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, str, float]]:
        qv = hash_tokens(tokenize(query), self.dim)
        scores = self.doc_vecs @ qv
        idx = np.argsort(-scores)[:top_k]
        return [(self.doc_ids[i], self.doc_texts[i], float(scores[i])) for i in idx]
