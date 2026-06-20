"""Offline embedding: hashing bag-of-words -> fixed-dim L2-normalised vector.

Stands in for a "small embedding model"; no network / model download needed.
cos(e(q), e(q')) is used by the semantic cache (Eq. 1 / Algorithm 1).
"""
from __future__ import annotations

import re
import zlib

import numpy as np

_TOK = re.compile(r"[a-z0-9]+")
# very small stop-list so paraphrases ("what is the" vs "what's") align better
_STOP = {"the", "a", "an", "is", "are", "of", "what", "whats", "who", "in",
         "do", "does", "to", "did", "was", "were", "which", "that", "s"}


def _h(s: str) -> int:
    return zlib.crc32(s.encode("utf-8"))   # deterministic across processes


def tokenize(text: str):
    return [t for t in _TOK.findall(text.lower()) if t not in _STOP]


def embed(text: str, dim: int = 2048) -> np.ndarray:
    """Word-level hashing trick -> L2-normalised vector. Word-only (no char-grams)
    keeps the toy retrieval/cosine clean and deterministic across processes."""
    v = np.zeros(dim, dtype=np.float32)
    for tok in tokenize(text):
        v[_h("w_" + tok) % dim] += 1.0
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))   # vectors are already L2-normalised
