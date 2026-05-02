from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

from chunking import Chunk
from model import BiEncoder, Vocab, batch_encode


def simple_tokenize(text: str) -> List[str]:
    return [t.strip(".,:;!?()\"'“”").lower() for t in text.split() if t.strip()]


@dataclass(frozen=True)
class RetrievalMetrics:
    precision: float
    recall: float
    iou: float


def token_metrics(reference: str, retrieved: str) -> RetrievalMetrics:
    ref_tokens = simple_tokenize(reference)
    ret_tokens = simple_tokenize(retrieved)

    ref_set = set(ref_tokens)
    ret_set = set(ret_tokens)
    if not ret_set:
        return RetrievalMetrics(precision=0.0, recall=0.0, iou=0.0)

    intersection = ref_set & ret_set
    union = ref_set | ret_set

    precision = len(intersection) / max(1, len(ret_set))
    recall = len(intersection) / max(1, len(ref_set))
    iou = len(intersection) / max(1, len(union))
    return RetrievalMetrics(precision=precision, recall=recall, iou=iou)


def build_index_embeddings(
    model: BiEncoder,
    vocab: Vocab,
    chunks: Sequence[Chunk],
    *,
    device: str = "cpu",
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        ids = batch_encode([c.text for c in chunks], vocab).to(device)
        emb = model(ids)
    return emb.cpu()


def retrieve_topk(
    query: str,
    *,
    model: BiEncoder,
    vocab: Vocab,
    chunk_embeddings: torch.Tensor,
    chunks: Sequence[Chunk],
    k: int = 3,
    device: str = "cpu",
) -> List[Chunk]:
    model.eval()
    with torch.no_grad():
        q_emb = model(batch_encode([query], vocab).to(device)).cpu()  # [1, D]

    scores = (q_emb @ chunk_embeddings.T).squeeze(0).numpy()
    topk = np.argsort(-scores)[:k]
    return [chunks[i] for i in topk]


def evaluate_retrieval(
    *,
    model: BiEncoder,
    vocab: Vocab,
    chunks: Sequence[Chunk],
    queries: Sequence[tuple[str, str]],
    gold_reference: Dict[str, str],
    k: int = 3,
    device: str = "cpu",
) -> RetrievalMetrics:
    """Evaluate average token-level precision/recall/IoU.

    Args:
        queries: list of (query_id, query_text)
        gold_reference: query_id -> reference text span
    """

    chunk_emb = build_index_embeddings(model, vocab, chunks, device=device)

    precisions: List[float] = []
    recalls: List[float] = []
    ious: List[float] = []

    for qid, qtext in queries:
        retrieved_chunks = retrieve_topk(
            qtext,
            model=model,
            vocab=vocab,
            chunk_embeddings=chunk_emb,
            chunks=chunks,
            k=k,
            device=device,
        )
        retrieved_text = " ".join(c.text for c in retrieved_chunks)
        m = token_metrics(gold_reference[qid], retrieved_text)
        precisions.append(m.precision)
        recalls.append(m.recall)
        ious.append(m.iou)

    return RetrievalMetrics(
        precision=float(np.mean(precisions)),
        recall=float(np.mean(recalls)),
        iou=float(np.mean(ious)),
    )
