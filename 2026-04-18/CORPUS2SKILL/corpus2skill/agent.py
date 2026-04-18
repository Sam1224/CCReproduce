from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class SearchResult:
    doc_id: str
    score: float
    title: str
    snippet: str


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0][0])


class Corpus2SkillAgent:
    def __init__(self, *, index: dict, vectorizer, out_dir: Path):
        self._index = index
        self._vectorizer = vectorizer
        self._out_dir = out_dir

        self._node_embeddings = {
            node_id: np.asarray(vec, dtype=np.float32) for node_id, vec in index["node_embeddings"].items()
        }

    @classmethod
    def load(cls, out_dir: Path) -> "Corpus2SkillAgent":
        index = json.loads((out_dir / "index.json").read_text(encoding="utf-8"))
        vectorizer = joblib.load(out_dir / "vectorizer.joblib")
        return cls(index=index, vectorizer=vectorizer, out_dir=out_dir)

    def _embed(self, text: str) -> np.ndarray:
        return self._vectorizer.transform([text]).toarray()[0].astype(np.float32)

    def _pick_child(self, query_vec: np.ndarray, node: dict) -> dict:
        best_child = None
        best_score = -1.0
        for child in node.get("children", []):
            emb = self._node_embeddings.get(child["node_id"])
            if emb is None:
                continue
            score = _cos(query_vec, emb)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def _load_doc(self, doc_id: str) -> tuple[str, str]:
        path = self._out_dir / "docs" / f"{doc_id}.md"
        text = path.read_text(encoding="utf-8")
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        title = lines[0].lstrip("# ") if lines else doc_id
        snippet = " ".join(lines[1:4])[:240]
        return title, snippet

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        query_vec = self._embed(query)

        # Greedy navigation down the tree.
        node = self._index["root"]
        while True:
            child = self._pick_child(query_vec, node)
            if child is None:
                break
            if not child.get("children"):
                node = child
                break
            node = child

        # Score docs inside the selected node.
        doc_ids = node.get("doc_ids", [])
        scored: list[SearchResult] = []
        for doc_id in doc_ids:
            title, snippet = self._load_doc(doc_id)
            doc_vec = self._embed(title + "\n" + snippet)
            score = _cos(query_vec, doc_vec)
            scored.append(SearchResult(doc_id=doc_id, score=score, title=title, snippet=snippet))

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]
