from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple

from dataset import RequestCase
from evolution import StrategyConfig

TARGETS: Dict[Tuple[str, str], Dict[str, object]] = {
    ("live_stream", "gmv_growth"): {
        "mechanism": "ctr_boost",
        "params": [0.78, 0.42, 0.18, 0.30],
        "base": {"gmv": 2.6, "ctr": 1.2, "refund_safety": 0.12, "creator_fairness": 0.18},
    },
    ("short_video_feed", "creator_fairness"): {
        "mechanism": "creator_diversity",
        "params": [0.34, 0.24, 0.88, 0.34],
        "base": {"gmv": 1.3, "ctr": 0.7, "refund_safety": 0.16, "creator_fairness": 1.15},
    },
    ("search", "quality_compliance"): {
        "mechanism": "quality_guardrail",
        "params": [0.22, 0.30, 0.14, 0.94],
        "base": {"gmv": 1.55, "ctr": 0.62, "refund_safety": 0.92, "creator_fairness": 0.12},
    },
    ("shelf_recommendation", "gmv_growth"): {
        "mechanism": "price_sensitive_rerank",
        "params": [0.46, 0.82, 0.24, 0.58],
        "base": {"gmv": 2.5, "ctr": 0.95, "refund_safety": 0.28, "creator_fairness": 0.10},
    },
    ("search", "retention"): {
        "mechanism": "cold_start_explore",
        "params": [0.40, 0.30, 0.78, 0.36],
        "base": {"gmv": 1.45, "ctr": 0.86, "refund_safety": 0.18, "creator_fairness": 0.72},
    },
}

MECHANISM_COMPAT = {
    "ctr_boost": {"ctr_boost": 1.0, "cvr_calibration": 0.5, "price_sensitive_rerank": 0.4},
    "creator_diversity": {"creator_diversity": 1.0, "cold_start_explore": 0.8, "quality_guardrail": 0.4},
    "quality_guardrail": {"quality_guardrail": 1.0, "price_sensitive_rerank": 0.6, "cvr_calibration": 0.4},
    "price_sensitive_rerank": {"price_sensitive_rerank": 1.0, "cvr_calibration": 0.7, "ctr_boost": 0.5},
    "cold_start_explore": {"cold_start_explore": 1.0, "creator_diversity": 0.8, "ctr_boost": 0.3},
}


class ABSimulator:
    def __init__(self, seed: int = 7) -> None:
        self.seed = seed

    def run(self, request: RequestCase, strategy: StrategyConfig, step: int) -> Dict[str, float]:
        target = TARGETS[(request.scenario, request.objective)]
        target_params: List[float] = target["params"]  # type: ignore[assignment]
        base: Dict[str, float] = target["base"]  # type: ignore[assignment]
        compat = MECHANISM_COMPAT[target["mechanism"]].get(strategy.mechanism, 0.2)
        distance = sum(abs(left - right) for left, right in zip(strategy.params, target_params)) / len(target_params)
        rng = random.Random(self.seed + hash((request.request_id, strategy.mechanism, step)) % 100000)
        drift = max(0.0, 1.0 - 0.10 * step)
        gmv = base["gmv"] * compat * drift + 0.8 * strategy.params[0] + 0.4 * strategy.params[1] - 1.4 * distance + rng.uniform(-0.08, 0.08)
        ctr = base["ctr"] * compat + 0.6 * strategy.params[0] + 0.25 * strategy.params[2] - 0.6 * distance + rng.uniform(-0.05, 0.05)
        refund_safety = base["refund_safety"] + 0.85 * strategy.params[3] - 0.45 * strategy.params[0] - 0.22 * strategy.params[2] + rng.uniform(-0.05, 0.05)
        creator_fairness = base["creator_fairness"] + 0.90 * strategy.params[2] + 0.25 * strategy.params[3] - 0.18 * strategy.params[1] + rng.uniform(-0.06, 0.06)
        confidence = min(0.99, 0.55 + 0.25 * math.tanh(request.traffic / 120000.0) + 0.04 * step)
        return {
            "gmv": round(gmv, 4),
            "ctr": round(ctr, 4),
            "refund_safety": round(refund_safety, 4),
            "creator_fairness": round(creator_fairness, 4),
            "confidence": round(confidence, 4),
        }
