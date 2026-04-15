from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class DenseRetriever:
    def __init__(self, model_name: str) -> None:
        self._model = SentenceTransformer(model_name)
        self._index: faiss.IndexFlatIP | None = None
        self._doc_ids: list[str] = []
        self._emb: np.ndarray | None = None

    def build(self, docs: Sequence[tuple[str, str]]) -> None:
        self._doc_ids = [doc_id for doc_id, _ in docs]
        texts = [text for _, text in docs]

        emb = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        emb = np.asarray(emb, dtype=np.float32)

        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)

        self._index = index
        self._emb = emb

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        if self._index is None:
            raise RuntimeError("index is not built")

        q_emb = self._model.encode([query], normalize_embeddings=True, show_progress_bar=False)
        q_emb = np.asarray(q_emb, dtype=np.float32)

        scores, ids = self._index.search(q_emb, top_k)

        out: list[tuple[str, float]] = []
        for idx, score in zip(ids[0].tolist(), scores[0].tolist()):
            if idx < 0:
                continue
            out.append((self._doc_ids[idx], float(score)))
        return out
