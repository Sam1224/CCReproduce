from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from data import Document, QueryExample, build_toy_corpus, domain_entities, relation_lexicon
from model import Chunk, FixedSizeChunker, SemanticChunkConfig, SemanticChunker, lexical_overlap
from train import rank_chunks, train_scorer


def build_semantic_chunks(documents: Sequence[Document]) -> List[Chunk]:
    chunker = SemanticChunker(SemanticChunkConfig(), relation_lexicon(), domain_entities())
    chunks: List[Chunk] = []
    for document in documents:
        chunks.extend(chunker.chunk_document(document.doc_id, document.text))
    return chunks


def build_fixed_chunks(documents: Sequence[Document]) -> List[Chunk]:
    chunker = FixedSizeChunker(window_tokens=12)
    chunks: List[Chunk] = []
    for document in documents:
        chunks.extend(chunker.chunk_document(document.doc_id, document.text))
    return chunks


def evaluate_top1(queries: Sequence[QueryExample], ranked: Dict[str, List[Tuple[Chunk, float]]]) -> float:
    correct = 0
    for query in queries:
        top_chunk = ranked[query.query][0][0]
        correct += int(top_chunk.doc_id == query.expected_doc_id and query.expected_object.lower() in top_chunk.text.lower())
    return correct / len(queries)


def fixed_rank(query: str, chunks: Sequence[Chunk]) -> List[Tuple[Chunk, float]]:
    return sorted(((chunk, lexical_overlap(query, chunk.text)) for chunk in chunks), key=lambda item: item[1], reverse=True)


def run_experiment() -> Dict[str, float]:
    documents, queries = build_toy_corpus()
    semantic_chunks = build_semantic_chunks(documents)
    fixed_chunks = build_fixed_chunks(documents)
    model, vocab = train_scorer(documents, queries, semantic_chunks)

    semantic_ranked = {query.query: rank_chunks(model, vocab, query.query, semantic_chunks) for query in queries}
    fixed_ranked = {query.query: fixed_rank(query.query, fixed_chunks) for query in queries}

    return {
        "semantic_top1": evaluate_top1(queries, semantic_ranked),
        "fixed_top1": evaluate_top1(queries, fixed_ranked),
        "semantic_chunks": float(len(semantic_chunks)),
        "fixed_chunks": float(len(fixed_chunks)),
    }


def main() -> None:
    metrics = run_experiment()
    print("Configurable Semantic Chunking toy evaluation")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    assert metrics["semantic_top1"] >= metrics["fixed_top1"]


if __name__ == "__main__":
    main()
