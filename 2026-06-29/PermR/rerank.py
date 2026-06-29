from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np


def dcg(rels: Sequence[float]) -> float:
    return float(sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(rels)))


def ndcg_at_k(rels: Sequence[float], k: int) -> float:
    rels = list(rels)[:k]
    ideal = sorted(rels, reverse=True)
    denom = dcg(ideal)
    if denom <= 1e-12:
        return 0.0
    return dcg(rels) / denom


@dataclass
class ConstraintConfig:
    k: int = 5
    max_rel_drop: float = 0.01  # allow <=1% ndcg@k drop vs baseline
    max_risk_increase: float = 0.02  # allow <=0.02 avg risk@k increase vs baseline


@dataclass
class RerankResult:
    perm: List[int]
    revenue_at_k: float
    ndcg_at_k: float
    avg_risk_at_k: float


def evaluate_perm(
    perm: Sequence[int],
    rel: Sequence[float],
    revenue: Sequence[float],
    risk: Sequence[float],
    k: int,
) -> RerankResult:
    top = list(perm)[:k]
    rel_top = [rel[i] for i in top]
    risk_top = [risk[i] for i in top]

    # Production monetization is position-sensitive. To avoid local-search plateaus
    # (e.g., swapping below top-K does not change a top-K sum), we use a simple
    # exposure discount over the *full* permutation as the objective.
    discounts = 1.0 / np.log2(np.arange(len(perm)) + 2)
    revenue_obj = float(sum(revenue[item] * discounts[pos] for pos, item in enumerate(perm)))

    return RerankResult(
        perm=list(perm),
        revenue_at_k=revenue_obj,
        ndcg_at_k=ndcg_at_k(rel_top, k),
        avg_risk_at_k=float(np.mean(risk_top)),
    )


def is_feasible(
    cand: RerankResult,
    base: RerankResult,
    cfg: ConstraintConfig,
) -> bool:
    if cand.ndcg_at_k < base.ndcg_at_k * (1.0 - cfg.max_rel_drop):
        return False
    if cand.avg_risk_at_k > base.avg_risk_at_k + cfg.max_risk_increase:
        return False
    return True


def baseline_rank(rel_score: Sequence[float]) -> List[int]:
    return sorted(range(len(rel_score)), key=lambda i: float(rel_score[i]), reverse=True)


def ilp_exact_by_enumeration(
    base_perm: Sequence[int],
    rel: Sequence[float],
    revenue: Sequence[float],
    risk: Sequence[float],
    cfg: ConstraintConfig,
) -> RerankResult:
    """Exact ILP solution by brute-force enumeration.

    The paper uses an ILP solver; to keep this repro dependency-free and runnable,
    we enumerate permutations. This is only feasible for small N (<=9).
    """

    n = len(rel)
    if n > 9:
        raise ValueError("Enumeration is exponential; set num_items<=9")

    base = evaluate_perm(base_perm, rel, revenue, risk, cfg.k)

    best: RerankResult | None = None
    for perm in itertools.permutations(range(n)):
        cand = evaluate_perm(perm, rel, revenue, risk, cfg.k)
        if not is_feasible(cand, base, cfg):
            continue
        if best is None or cand.revenue_at_k > best.revenue_at_k:
            best = cand

    return best or base


def _constraint_violation(cand: RerankResult, base: RerankResult, cfg: ConstraintConfig) -> float:
    """Positive means violated; 0 means feasible.

    We combine relevance and risk violations into one scalar to guide repairs.
    """

    rel_floor = base.ndcg_at_k * (1.0 - cfg.max_rel_drop)
    rel_violation = max(0.0, rel_floor - cand.ndcg_at_k)

    risk_ceiling = base.avg_risk_at_k + cfg.max_risk_increase
    risk_violation = max(0.0, cand.avg_risk_at_k - risk_ceiling)

    # Normalize to roughly comparable scales.
    return rel_violation + risk_violation


def permr_rerank(
    base_perm: Sequence[int],
    rel: Sequence[float],
    revenue: Sequence[float],
    risk: Sequence[float],
    cfg: ConstraintConfig,
    max_steps: int = 2000,
) -> RerankResult:
    """A lightweight neighbor-swap reranker inspired by PermR.

    High-level loop:
      1) If constraints violated, do a best neighbor swap to reduce violation.
      2) If constraints satisfied, do a best neighbor swap to improve revenue.

    This is intentionally simple and readable; it matches the paper's spirit of
    permutation-based constrained reranking under latency constraints.
    """

    perm = list(base_perm)
    base = evaluate_perm(base_perm, rel, revenue, risk, cfg.k)

    current = evaluate_perm(perm, rel, revenue, risk, cfg.k)

    for _ in range(max_steps):
        violated = not is_feasible(current, base, cfg)

        best_swap: Tuple[int, int] | None = None
        best_score: float | None = None
        best_after: RerankResult | None = None

        for i in range(len(perm) - 1):
            cand_perm = perm.copy()
            cand_perm[i], cand_perm[i + 1] = cand_perm[i + 1], cand_perm[i]
            cand = evaluate_perm(cand_perm, rel, revenue, risk, cfg.k)

            if violated:
                # Repair mode: prioritize reducing constraint violation; tie-break by revenue.
                vio = _constraint_violation(cand, base, cfg)
                score = -vio * 1e3 + cand.revenue_at_k
            else:
                # Improve mode: only consider feasible swaps and maximize revenue.
                if not is_feasible(cand, base, cfg):
                    continue
                score = cand.revenue_at_k

            if best_score is None or score > best_score:
                best_score = score
                best_swap = (i, i + 1)
                best_after = cand

        if best_swap is None or best_after is None:
            break

        # Stop if no improvement in the current regime.
        if not violated and best_after.revenue_at_k <= current.revenue_at_k + 1e-9:
            break

        perm[best_swap[0]], perm[best_swap[1]] = perm[best_swap[1]], perm[best_swap[0]]
        current = best_after

        if is_feasible(current, base, cfg) and current.revenue_at_k <= base.revenue_at_k:
            # early exit if we end up worse than baseline while already feasible
            break

    return current
