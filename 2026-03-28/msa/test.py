from __future__ import annotations

import torch

from dataset import MemoryQADataset, collate
from model import MSAConfig, MSAModel


def test_forward() -> None:
    cfg = MSAConfig(n_docs=4, doc_len=16, chunk_size=4, top_k=2)
    ds = MemoryQADataset(n_samples=4, n_docs=cfg.n_docs, doc_len=cfg.doc_len, vocab_size=cfg.vocab_size, seed=0)
    b = collate([ds[i] for i in range(4)])

    model = MSAModel(cfg)
    logits, scores = model(b.docs, b.query)
    assert logits.shape == (4, cfg.vocab_size)
    assert scores.shape == (4, cfg.n_docs)


if __name__ == "__main__":
    test_forward()
    print("ok")
