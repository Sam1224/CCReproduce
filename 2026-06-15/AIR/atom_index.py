"""
AIR - Atom Index Builder

Builds a FAISS index over intent atom embeddings for efficient online retrieval.

Paper (§3.2): Intent atoms are encoded into dense vectors using a text encoder.
During online serving, the system retrieves the k most relevant atoms given
the current request context, achieving ~400× latency reduction vs. online LLM.
"""
import json
import os
from typing import Optional

import numpy as np

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    print("[WARN] faiss not installed, falling back to numpy brute-force index.")


class AtomIndex:
    """
    Dense index over intent atoms.

    In production (per paper):
    - Text encoder: e.g., sentence-transformers or fine-tuned bi-encoder
    - Index: FAISS IVF for billion-scale retrieval
    - Latency: sub-millisecond per query (vs. seconds for online LLM)

    This toy uses random 128-dim embeddings to simulate the index structure.
    """

    def __init__(self, embed_dim: int = 128):
        self.embed_dim = embed_dim
        self.atom_data: list[dict] = []
        self.embeddings: Optional[np.ndarray] = None
        self._faiss_index = None

    def _encode_text(self, text: str) -> np.ndarray:
        """
        Toy text encoder — simulates a bi-encoder producing 128-dim vectors.

        In production, replace with sentence-transformers or a fine-tuned encoder.
        Formula (paper §3.2): e(a) = Encoder(a.text)  where a is an intent atom.
        """
        # Deterministic hash-based fake embedding for reproducibility
        np.random.seed(hash(text) % (2**31))
        vec = np.random.randn(self.embed_dim).astype(np.float32)
        vec /= np.linalg.norm(vec) + 1e-8
        return vec

    def build(self, atoms: list[dict]) -> None:
        """Build the index from a list of intent atom dicts."""
        self.atom_data = atoms
        embeddings = np.stack([self._encode_text(a["text"]) for a in atoms])
        self.embeddings = embeddings

        if HAS_FAISS:
            index = faiss.IndexFlatIP(self.embed_dim)  # inner product (cosine after norm)
            faiss.normalize_L2(embeddings)
            index.add(embeddings)
            self._faiss_index = index
        print(f"Built atom index with {len(atoms)} atoms, dim={self.embed_dim}.")

    def search(self, query_text: str, top_k: int = 10) -> list[dict]:
        """
        Retrieve top-k intent atoms for a query.

        Paper §3.3: online composition retrieves atoms matching the current
        user context and composes them into a unified intent representation.
        """
        query_vec = self._encode_text(query_text).reshape(1, -1)

        if HAS_FAISS and self._faiss_index is not None:
            faiss.normalize_L2(query_vec)
            scores, indices = self._faiss_index.search(query_vec, top_k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue
                atom = dict(self.atom_data[idx])
                atom["retrieval_score"] = float(score)
                results.append(atom)
        else:
            # Brute-force numpy fallback
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8
            normed = self.embeddings / norms
            query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
            sims = (normed @ query_norm.T).flatten()
            top_idx = np.argsort(-sims)[:top_k]
            results = []
            for idx in top_idx:
                atom = dict(self.atom_data[idx])
                atom["retrieval_score"] = float(sims[idx])
                results.append(atom)

        return results

    def save(self, path: str) -> None:
        """Save index state (embeddings + metadata)."""
        np.save(path + ".npy", self.embeddings)
        with open(path + ".meta.json", "w") as f:
            json.dump(self.atom_data, f)
        print(f"Saved atom index to {path}.")

    @classmethod
    def load(cls, path: str) -> "AtomIndex":
        """Load a previously built index."""
        idx = cls()
        idx.embeddings = np.load(path + ".npy")
        with open(path + ".meta.json") as f:
            idx.atom_data = json.load(f)
        idx.embed_dim = idx.embeddings.shape[1]
        if HAS_FAISS:
            faiss_idx = faiss.IndexFlatIP(idx.embed_dim)
            normed = idx.embeddings.copy()
            faiss.normalize_L2(normed)
            faiss_idx.add(normed)
            idx._faiss_index = faiss_idx
        print(f"Loaded atom index with {len(idx.atom_data)} atoms.")
        return idx


if __name__ == "__main__":
    import sys
    atoms_path = sys.argv[1] if len(sys.argv) > 1 else "data/atoms.json"
    index_path = sys.argv[2] if len(sys.argv) > 2 else "data/atom_index"

    with open(atoms_path) as f:
        atoms = json.load(f)

    idx = AtomIndex(embed_dim=128)
    idx.build(atoms)
    idx.save(index_path)

    # Quick sanity check
    results = idx.search("user interested in cooking and kitchen products", top_k=3)
    print("Sample retrieval results:")
    for r in results:
        print(f"  [{r['retrieval_score']:.3f}] {r['text']} (cat={r['category']})")
