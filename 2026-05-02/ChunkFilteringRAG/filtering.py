from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from chunking import Chunk


_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


def normalize_for_exact_dedup(text: str) -> str:
    tokens = _WORD_RE.findall(text.lower())
    return " ".join(tokens)


def extract_pseudo_entities(text: str) -> List[str]:
    """A lightweight entity proxy.

    In the original paper, an NER-based strategy is evaluated.
    To keep this reproduction dependency-light, we approximate entities as
    capitalized words (plus alphanumerics) from the raw text.
    """

    candidates = re.findall(r"\b[A-Z][A-Za-z0-9-]{2,}\b", text)
    return [c.lower() for c in candidates]


def jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


@dataclass(frozen=True)
class FilterStats:
    original_count: int
    kept_count: int

    @property
    def reduction_ratio(self) -> float:
        if self.original_count == 0:
            return 0.0
        return 1.0 - self.kept_count / self.original_count


def exact_dedup(chunks: Iterable[Chunk]) -> Tuple[List[Chunk], FilterStats]:
    seen: Dict[str, str] = {}
    kept: List[Chunk] = []
    total = 0

    for chunk in chunks:
        total += 1
        key = normalize_for_exact_dedup(chunk.text)
        if key in seen:
            continue
        seen[key] = chunk.chunk_id
        kept.append(chunk)

    return kept, FilterStats(original_count=total, kept_count=len(kept))


def entity_jaccard_filter(chunks: Sequence[Chunk], *, threshold: float = 0.7) -> Tuple[List[Chunk], FilterStats]:
    """Keep the first chunk among near-duplicates under entity-set Jaccard.

    This is intentionally simple: O(n^2) on the chunk count.
    For real deployments, you'd bucket by doc/topic or use LSH.
    """

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be within [0, 1]")

    entities = [extract_pseudo_entities(c.text) for c in chunks]

    kept: List[Chunk] = []
    kept_entities: List[List[str]] = []

    for chunk, ents in zip(chunks, entities, strict=True):
        is_redundant = False
        for kept_ents in kept_entities:
            if jaccard(ents, kept_ents) >= threshold:
                is_redundant = True
                break
        if not is_redundant:
            kept.append(chunk)
            kept_entities.append(ents)

    return kept, FilterStats(original_count=len(chunks), kept_count=len(kept))


def cosine_similarity_matrix(x: np.ndarray) -> np.ndarray:
    x_norm = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    return x_norm @ x_norm.T
