from __future__ import annotations

from data import build_toy_corpus
from test import build_semantic_chunks, run_experiment
from train import rank_chunks, train_scorer


def main() -> None:
    documents, queries = build_toy_corpus()
    chunks = build_semantic_chunks(documents)
    model, vocab = train_scorer(documents, queries, chunks)

    print("Top chunks by query")
    for query in queries:
        top_chunk, score = rank_chunks(model, vocab, query.query, chunks)[0]
        print(f"\nQ: {query.query}")
        print(f"doc={top_chunk.doc_id} relation={top_chunk.relation} trigger={top_chunk.trigger} score={score:.3f}")
        print(top_chunk.text)

    print("\nAggregate metrics")
    for name, value in run_experiment().items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
