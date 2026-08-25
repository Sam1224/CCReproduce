from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from data import EvalRecord, ensure_toy_data, load_eval_records, text_to_ids
from model import BiEncoderConfig, DenseRetriever


def dcg(grades: Iterable[int], k: int) -> float:
    total = 0.0
    for index, grade in enumerate(list(grades)[:k], start=1):
        gain = (2**grade) - 1
        total += gain / np.log2(index + 1)
    return total


@torch.no_grad()
def evaluate(model: DenseRetriever, records: list[EvalRecord], *, k: int = 10, vocab_size: int = 4096, max_len: int = 20) -> dict[str, float]:
    model.eval()
    ndcg_scores: list[float] = []
    recall_scores: list[float] = []
    embarrassing_scores: list[float] = []
    tail_ndcg: list[float] = []
    head_ndcg: list[float] = []

    for record in records:
        query_ids = torch.tensor([text_to_ids(record.query_text, max_len=max_len, vocab_size=vocab_size)], dtype=torch.long)
        doc_ids = torch.tensor(
            [text_to_ids(candidate.item_text, max_len=max_len, vocab_size=vocab_size) for candidate in record.candidates],
            dtype=torch.long,
        )
        scores = model.score_matrix(query_ids, doc_ids).squeeze(0).cpu().numpy()
        order = list(np.argsort(-scores))
        ranked_grades = [record.candidates[index].grade for index in order]

        ideal_grades = sorted((candidate.grade for candidate in record.candidates), reverse=True)
        denom = dcg(ideal_grades, k)
        ndcg = dcg(ranked_grades, k) / denom if denom > 0 else 0.0
        recall = float(any(grade >= 3 for grade in ranked_grades[:k]))
        embarrassing = float(ranked_grades[0] == 0)

        ndcg_scores.append(ndcg)
        recall_scores.append(recall)
        embarrassing_scores.append(embarrassing)
        if record.head_tail == "tail":
            tail_ndcg.append(ndcg)
        else:
            head_ndcg.append(ndcg)

    return {
        f"ndcg@{k}": float(np.mean(ndcg_scores)),
        f"recall@{k}": float(np.mean(recall_scores)),
        "embarrassing@1": float(np.mean(embarrassing_scores)),
        f"tail_ndcg@{k}": float(np.mean(tail_ndcg)) if tail_ndcg else 0.0,
        f"head_ndcg@{k}": float(np.mean(head_ndcg)) if head_ndcg else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="toy_data")
    parser.add_argument("--checkpoint", default="checkpoints/scaling_dense_retrieval.pt")
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topk-per-channel", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = ensure_toy_data(args.data_dir, seed=args.seed, topk_per_channel=args.topk_per_channel)
    records = load_eval_records(data_dir, args.split)
    state = torch.load(args.checkpoint, map_location="cpu")
    meta = state["meta"]
    cfg = BiEncoderConfig(vocab_size=meta["vocab_size"], d_model=meta["d_model"], max_len=meta["max_len"])
    model = DenseRetriever(cfg)
    model.load_state_dict(state["state"])
    metrics = evaluate(model, records, k=args.k, vocab_size=cfg.vocab_size, max_len=cfg.max_len)
    print(json.dumps({"split": args.split, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
