import argparse
import json
from pathlib import Path

import torch

from data import decision_swipes, make_splits
from model import SMEOPipeline


def flatten_rank_input(session, prefix):
    from data import _prefix_state  # local reuse for compactness

    state = _prefix_state(session.assets, prefix, session.purchase_goal)
    remaining = [i for i in range(len(session.assets)) if i not in prefix]
    ordered = remaining + [i for i in range(len(session.assets)) if i in prefix]
    flat = list(state)
    for idx in ordered:
        mask = [0.0] if idx in prefix else [1.0]
        flat.extend(session.assets[idx] + mask)
    return torch.tensor(flat, dtype=torch.float32).unsqueeze(0), ordered


def rollout(model, session):
    prefix = []
    while len(prefix) < len(session.assets):
        x, ordered = flatten_rank_input(session, prefix)
        logits = model.ranker(x)
        choice = ordered[int(logits.argmax(dim=-1).item())]
        if choice in prefix:
            break
        prefix.append(choice)
    return prefix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts")
    args = parser.parse_args()

    artifacts = Path(args.artifacts)
    pipeline = SMEOPipeline()
    pipeline.utility.load_state_dict(torch.load(artifacts / "utility.pt", map_location="cpu"))
    pipeline.ranker.load_state_dict(torch.load(artifacts / "ranker.pt", map_location="cpu"))
    pipeline.eval()

    splits = make_splits()
    test_sessions = splits["test_sessions"]

    learned_swipes = []
    baseline_swipes = []
    exact_prefix_hits = 0

    for session in test_sessions:
        pred_order = rollout(pipeline, session)
        learned_swipes.append(decision_swipes(pred_order, session.assets))
        baseline_swipes.append(decision_swipes(session.baseline_order, session.assets))
        if pred_order[:3] == session.best_order[:3]:
            exact_prefix_hits += 1

    report = {
        "learned_avg_swipes": round(sum(learned_swipes) / len(learned_swipes), 3),
        "baseline_avg_swipes": round(sum(baseline_swipes) / len(baseline_swipes), 3),
        "swipe_reduction_pct": round((1 - (sum(learned_swipes) / sum(baseline_swipes))) * 100, 2),
        "top3_prefix_match": round(exact_prefix_hits / len(test_sessions), 4),
    }

    with open(artifacts / "test_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
