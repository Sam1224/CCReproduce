from __future__ import annotations

import argparse

import torch

from data import evaluate_policy
from model import ExperienceOrchestrator, NaiveBaseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="eo_toy.pt")
    parser.add_argument("--eval-sessions", type=int, default=384)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.checkpoint, map_location=device)

    model = ExperienceOrchestrator(
        input_dim=payload["config"]["input_dim"],
        hidden_dim=payload["config"]["hidden_dim"],
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()

    max_turns = int(payload["config"]["max_turns"])
    eval_seed = int(payload["config"]["seed"]) + 500 if args.seed is None else args.seed + 500

    baseline = NaiveBaseline()
    baseline_metrics = evaluate_policy(
        baseline,
        num_sessions=args.eval_sessions,
        max_turns=max_turns,
        seed=eval_seed,
        device=None,
    )
    eo_metrics = evaluate_policy(
        model,
        num_sessions=args.eval_sessions,
        max_turns=max_turns,
        seed=eval_seed,
        device=device,
    )
    delta = {key: eo_metrics[key] - baseline_metrics[key] for key in eo_metrics}

    print(
        "[naive] "
        f"advisor_contact_rate={baseline_metrics['advisor_contact_rate']:.4f} "
        f"genuine_contact_rate={baseline_metrics['genuine_contact_rate']:.4f} "
        f"avg_resistance_drop={baseline_metrics['avg_resistance_drop']:.4f}"
    )
    print(
        "[eo] "
        f"advisor_contact_rate={eo_metrics['advisor_contact_rate']:.4f} "
        f"genuine_contact_rate={eo_metrics['genuine_contact_rate']:.4f} "
        f"avg_resistance_drop={eo_metrics['avg_resistance_drop']:.4f}"
    )
    print(
        "[delta] "
        f"advisor_contact_rate={delta['advisor_contact_rate']:+.4f} "
        f"genuine_contact_rate={delta['genuine_contact_rate']:+.4f} "
        f"avg_resistance_drop={delta['avg_resistance_drop']:+.4f}"
    )


if __name__ == "__main__":
    main()
