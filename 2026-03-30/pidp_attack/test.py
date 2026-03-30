from __future__ import annotations

from retriever import Retriever
from toy_data import build_toy_corpus


def test_retriever() -> None:
    r = Retriever(dim=64)
    corpus = build_toy_corpus()
    r.build([(d.doc_id, d.text) for d in corpus])
    hits = r.search("return window", top_k=2)
    assert len(hits) == 2


if __name__ == "__main__":
    test_retriever()
    print("ok")
