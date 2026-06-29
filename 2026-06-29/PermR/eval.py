from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from data import SyntheticRerankDataset
from model import ScoringModel
from rerank import ConstraintConfig, baseline_rank, evaluate_perm, ilp_exact_by_enumeration, permr_rerank


def _load_model(ckpt_path: Path) -> ScoringModel:
    payload = torch.load(ckpt_path, map_location="cpu")
    model = ScoringModel(feature_dim=payload["feature_dim"], hidden_dim=payload["hidden_dim"])
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    ckpt_path = base_dir / "checkpoints" / "scorer.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError("missing checkpoints/scorer.pt; run train.py first")

    model = _load_model(ckpt_path)

    dataset = SyntheticRerankDataset(num_queries=200, num_items=8, feature_dim=16, seed=2026)
    # Make constraints tight to mimic production: baseline ranking is strong,
    # and the feasible space is near the baseline.
    cfg = ConstraintConfig(k=5, max_rel_drop=0.002, max_risk_increase=0.005)

    sums = {
        "baseline_rev": 0.0,
        "permr_rev": 0.0,
        "ilp_rev": 0.0,
        "baseline_ndcg": 0.0,
        "permr_ndcg": 0.0,
        "ilp_ndcg": 0.0,
        "baseline_risk": 0.0,
        "permr_risk": 0.0,
        "ilp_risk": 0.0,
    }

    n = 0
    for q in dataset:
        with torch.no_grad():
            rel_logit, _ = model(q.features.unsqueeze(0).float())
            rel_score = rel_logit.squeeze(0).numpy()

        base_perm = baseline_rank(rel_score)

        rel = q.rel_label.numpy().tolist()
        revenue = q.revenue_label.numpy().tolist()
        risk = q.risk_label.numpy().tolist()

        base_res = evaluate_perm(base_perm, rel, revenue, risk, cfg.k)
        permr_res = permr_rerank(base_perm, rel, revenue, risk, cfg)
        ilp_res = ilp_exact_by_enumeration(base_perm, rel, revenue, risk, cfg)

        for key, res in [
            ("baseline", base_res),
            ("permr", permr_res),
            ("ilp", ilp_res),
        ]:
            sums[f"{key}_rev"] += res.revenue_at_k
            sums[f"{key}_ndcg"] += res.ndcg_at_k
            sums[f"{key}_risk"] += res.avg_risk_at_k

        n += 1

    def avg(x: float) -> float:
        return x / max(1, n)

    baseline_rev = avg(sums["baseline_rev"])
    permr_rev = avg(sums["permr_rev"])
    ilp_rev = avg(sums["ilp_rev"])

    print("=== PermR toy reproduction (avg over queries) ===")
    print(f"Queries: {n}  Items/query: {dataset.num_items}  K={cfg.k}")
    print("--- Objective (discounted revenue; higher better) ---")
    print(f"baseline: {baseline_rev:.3f}")
    print(f"PermR   : {permr_rev:.3f}  (gain {permr_rev/baseline_rev-1:+.2%})")
    print(f"ILP(opt): {ilp_rev:.3f}  (gain {ilp_rev/baseline_rev-1:+.2%})")
    print(f"PermR / ILP revenue-gain ratio: {(permr_rev-baseline_rev)/(ilp_rev-baseline_rev+1e-12):.2%}")

    print("--- NDCG@K (constraint) ---")
    print(f"baseline: {avg(sums['baseline_ndcg']):.4f}")
    print(f"PermR   : {avg(sums['permr_ndcg']):.4f}")
    print(f"ILP(opt): {avg(sums['ilp_ndcg']):.4f}")

    print("--- AvgRisk@K (constraint) ---")
    print(f"baseline: {avg(sums['baseline_risk']):.4f}")
    print(f"PermR   : {avg(sums['permr_risk']):.4f}")
    print(f"ILP(opt): {avg(sums['ilp_risk']):.4f}")


if __name__ == "__main__":
    main()
