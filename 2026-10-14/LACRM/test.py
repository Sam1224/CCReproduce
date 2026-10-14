from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from data import MarketConfig, ToyMarket, group_ground_truth, relevance_labels
from model import DualProfileEncoder, LACRMConfig, LambdaRankReranker, VIEW_NAMES
from train import build_candidate_bank, make_pair_features, market_to_tensors


ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_market(payload) -> ToyMarket:
    return ToyMarket(
        category=payload["category"],
        brand=payload["brand"],
        intent=payload["intent"],
        style=payload["style"],
        audience=payload["audience"],
        price=payload["price"],
        channel=payload["channel"],
        merchant_tier=payload["merchant_tier"],
        festival_active=payload["festival_active"],
        behavior=payload["behavior"],
        train_view_scores=payload["train_view_scores"],
        eval_view_scores=payload["eval_view_scores"],
        train_scores=payload["train_scores"],
        eval_scores=payload["eval_scores"],
    )


def precision_at_k(pred: np.ndarray, truth: np.ndarray, k: int) -> float:
    if len(pred) == 0:
        return 0.0
    pred_k = pred[:k]
    return float(np.isin(pred_k, truth).sum() / max(1, len(pred_k)))


def recall_at_k(pred: np.ndarray, truth: np.ndarray, k: int) -> float:
    if len(truth) == 0:
        return 0.0
    pred_k = pred[:k]
    return float(np.isin(pred_k, truth).sum() / max(1, len(truth)))


def ndcg_at_k(pred: np.ndarray, rel_lookup: dict[int, float], k: int) -> float:
    pred_k = pred[:k]
    if len(pred_k) == 0:
        return 0.0
    gains = np.array([rel_lookup.get(int(idx), 0.0) for idx in pred_k], dtype=np.float32)
    order = np.argsort(list(rel_lookup.values()))[::-1]
    ideal = np.array(sorted(rel_lookup.values(), reverse=True)[:k], dtype=np.float32)
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    dcg = float(np.sum((2.0 ** gains - 1.0) * discounts))
    idcg = float(np.sum((2.0 ** ideal - 1.0) * discounts[: len(ideal)]))
    return dcg / max(1e-6, idcg)


def evaluate(
    market: ToyMarket,
    candidates: list[np.ndarray],
    sem_scores: torch.Tensor,
    reranker_scores: list[np.ndarray],
    rerank_weight: float = 0.2,
):
    truth = group_ground_truth(market.eval_scores, k=20, threshold=0.25)

    semantic_rankings: list[np.ndarray] = []
    full_rankings: list[np.ndarray] = []
    rels = relevance_labels(market.eval_scores, candidates)
    for src, cand in enumerate(candidates):
        if cand.size == 0:
            semantic_rankings.append(cand)
            full_rankings.append(cand)
            continue
        sem = sem_scores[src, :, cand].max(dim=0).values.detach().cpu().numpy()
        semantic_rankings.append(cand[np.argsort(sem)[::-1]])
        # The paper's final stage is a behavioral alignment on top of semantic
        # retrieval, not a hard replacement of the semantic route. In the toy
        # implementation we therefore refine semantic scores with a small
        # LambdaRank-style residual.
        full_rankings.append(cand[np.argsort(sem + rerank_weight * reranker_scores[src])[::-1]])

    out = {}
    for name, rankings in [("semantic_only", semantic_rankings), ("lacrm_full", full_rankings)]:
        p10, r10, n10, n20 = [], [], [], []
        for src, pred in enumerate(rankings):
            gold = truth[src]
            rel_lookup = {int(c): float(r) for c, r in zip(candidates[src], rels[src])}
            p10.append(precision_at_k(pred, gold, 10))
            r10.append(recall_at_k(pred, gold, 10))
            n10.append(ndcg_at_k(pred, rel_lookup, 10))
            n20.append(ndcg_at_k(pred, rel_lookup, 20))
        out[name] = {
            "precision@10": round(float(np.mean(p10)), 4),
            "recall@10": round(float(np.mean(r10)), 4),
            "ndcg@10": round(float(np.mean(n10)), 4),
            "ndcg@20": round(float(np.mean(n20)), 4),
        }
    return out


def main() -> None:
    ckpt = torch.load(ROOT / "lacrm_ckpt.pt", map_location=DEVICE, weights_only=False)
    market_cfg = MarketConfig(**ckpt["market_cfg"])
    model_cfg = LACRMConfig(**ckpt["model_cfg"])
    market = load_market(ckpt["market"])
    fields = market_to_tensors(market)

    encoder = DualProfileEncoder(model_cfg).to(DEVICE)
    encoder.load_state_dict(ckpt["encoder_state"])
    encoder.eval()

    sem_scores, hit, candidates = build_candidate_bank(encoder, fields, market)
    group_features, _ = make_pair_features(market, sem_scores, hit, candidates)

    input_dim = next(feat.shape[1] for feat in group_features if feat.numel() > 0)
    reranker = LambdaRankReranker(input_dim=input_dim, hidden_dim=model_cfg.rerank_hidden).to(DEVICE)
    reranker.load_state_dict(ckpt["reranker_state"])
    reranker.eval()

    reranker_scores = []
    with torch.no_grad():
        for feat in group_features:
            if feat.numel() == 0:
                reranker_scores.append(np.array([], dtype=np.float32))
            else:
                reranker_scores.append(reranker(feat).cpu().numpy())

    metrics = evaluate(market, candidates, sem_scores, reranker_scores)
    metrics["device"] = str(DEVICE)
    metrics["num_sources"] = int(market_cfg.n_advertisers)

    out_path = ROOT / "test_metrics.json"
    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
