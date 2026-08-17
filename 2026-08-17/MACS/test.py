from __future__ import annotations

from pathlib import Path

import torch

from data import benchmark_sessions, build_catalog
from model import ConstraintAwareRanker, MACSPipeline


def main() -> None:
    catalog = build_catalog()
    checkpoint = torch.load(Path(__file__).resolve().parent / "macs_toy.pt", map_location="cpu")
    model = ConstraintAwareRanker(checkpoint["query_dim"], checkpoint["candidate_dim"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    pipeline = MACSPipeline(model, catalog)

    total_cases = 0
    passed_cases = 0
    budget_compliance = 0
    exclusion_compliance = 0

    for turns in benchmark_sessions():
        pipeline.reset()
        for turn in turns:
            recommendations = pipeline.recommend(turn, top_k=3)
            if not recommendations:
                continue
            total_cases += 1
            top_item = recommendations[0]
            if pipeline.memory.budget_max == 0 or top_item.price <= pipeline.memory.budget_max + 1e-6:
                budget_compliance += 1
            if top_item.brand not in pipeline.memory.excluded_brands:
                exclusion_compliance += 1
            if recommendations:
                passed_cases += 1

    pass_rate = passed_cases / max(total_cases, 1)
    budget_rate = budget_compliance / max(total_cases, 1)
    exclusion_rate = exclusion_compliance / max(total_cases, 1)

    print({
        "pass_rate": round(pass_rate, 4),
        "budget_compliance": round(budget_rate, 4),
        "exclusion_compliance": round(exclusion_rate, 4),
        "evaluated_turns": total_cases,
    })


if __name__ == "__main__":
    main()
