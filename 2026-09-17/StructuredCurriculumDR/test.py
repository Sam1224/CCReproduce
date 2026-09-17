from __future__ import annotations

import os

import torch

from data import (
    build_cfg_from_ckpt,
    evaluate_retrieval,
    make_toy_catalog,
    make_toy_queries,
    train_valid_test_split,
)
from model import DualEncoder
from train import BASELINE_CKPT, CURRICULUM_CKPT


def load_payload(path: str, device: torch.device):
    payload = torch.load(path, weights_only=False, map_location=device)
    cfg = build_cfg_from_ckpt(payload)
    model = DualEncoder(vocab_size=len(payload["vocab"]), dim=payload["dim"]).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    return payload, cfg, model


def main() -> None:
    if not (os.path.exists(BASELINE_CKPT) and os.path.exists(CURRICULUM_CKPT)):
        raise SystemExit("missing checkpoints; run `python train.py` first")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_payload, cfg, baseline = load_payload(BASELINE_CKPT, device)
    cur_payload, _, curriculum = load_payload(CURRICULUM_CKPT, device)

    products = make_toy_catalog(cfg.n_products, seed=cfg.seed)
    queries = make_toy_queries(cfg.n_queries, seed=cfg.seed + 1)
    _, _, q_test = train_valid_test_split(queries, seed=cfg.seed)

    base_metrics = evaluate_retrieval(baseline, base_payload["vocab"], q_test, products, device=device, k=10)
    cur_metrics = evaluate_retrieval(curriculum, cur_payload["vocab"], q_test, products, device=device, k=10)

    print("=== Structured curriculum dense retrieval ===")
    print("baseline :", {k: round(v, 4) for k, v in base_metrics.items()})
    print("curriculum:", {k: round(v, 4) for k, v in cur_metrics.items()})
    print("expectation: curriculum improves ndcg/recall and reduces embarrassing@1 over the click-like baseline")


if __name__ == "__main__":
    main()
