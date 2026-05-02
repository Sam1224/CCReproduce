from __future__ import annotations

from dataclasses import dataclass

import torch

from chunking import chunk_corpus
from filtering import entity_jaccard_filter, exact_dedup
from model import BiEncoder, Vocab
from train import PairExample, train_biencoder
from toy_data import as_doc_tuples, build_gold_reference, build_toy_corpus
from eval_retrieval import evaluate_retrieval


@dataclass(frozen=True)
class ExperimentResult:
    name: str
    chunk_count: int
    reduction_ratio: float
    precision: float
    recall: float
    iou: float


def run() -> None:
    docs, toy_queries = build_toy_corpus()

    chunks = chunk_corpus(as_doc_tuples(docs), chunk_size=20, overlap=10)
    baseline_chunks = chunks

    vocab = Vocab.build([c.text for c in baseline_chunks] + [q.query for q in toy_queries])
    model = BiEncoder(vocab_size=len(vocab.token_to_id), dim=128)

    # Train on query -> gold doc span (toy setting)
    train_examples = [PairExample(query=q.query, positive=q.gold_span) for q in toy_queries]
    model = train_biencoder(model, vocab, train_examples, epochs=80, batch_size=3, lr=5e-3)

    gold_ref = build_gold_reference(toy_queries)
    query_pairs = [(q.query_id, q.query) for q in toy_queries]

    baseline_metrics = evaluate_retrieval(
        model=model,
        vocab=vocab,
        chunks=baseline_chunks,
        queries=query_pairs,
        gold_reference=gold_ref,
        k=3,
    )

    # Filtering pipeline: exact dedup -> entity-jaccard
    dedup_chunks, _ = exact_dedup(baseline_chunks)
    filtered_chunks, stats = entity_jaccard_filter(dedup_chunks, threshold=0.3)

    filtered_metrics = evaluate_retrieval(
        model=model,
        vocab=vocab,
        chunks=filtered_chunks,
        queries=query_pairs,
        gold_reference=gold_ref,
        k=3,
    )

    results = [
        ExperimentResult(
            name="baseline",
            chunk_count=len(baseline_chunks),
            reduction_ratio=0.0,
            precision=baseline_metrics.precision,
            recall=baseline_metrics.recall,
            iou=baseline_metrics.iou,
        ),
        ExperimentResult(
            name="filtered(entity_jaccard)",
            chunk_count=len(filtered_chunks),
            reduction_ratio=stats.reduction_ratio,
            precision=filtered_metrics.precision,
            recall=filtered_metrics.recall,
            iou=filtered_metrics.iou,
        ),
    ]

    print("== Chunk Filtering (toy reproduction) ==")
    for r in results:
        print(
            f"{r.name:>22} | chunks={r.chunk_count:>3} | reduction={r.reduction_ratio*100:>5.1f}% "
            f"| P={r.precision:.3f} R={r.recall:.3f} IoU={r.iou:.3f}"
        )


if __name__ == "__main__":
    torch.manual_seed(7)
    run()
