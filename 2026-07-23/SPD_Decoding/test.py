from __future__ import annotations

import argparse

import torch

from data import Catalog, create_dataloaders, slate_metrics
from model import GenerativeSlateModel, decode_greedy, stochastic_primal_dual_decode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/spd_decoding.pt")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def load_model(checkpoint_path: str, device: torch.device) -> tuple[GenerativeSlateModel, Catalog, int]:
    payload = torch.load(checkpoint_path, map_location=device)
    config = payload["config"]
    model = GenerativeSlateModel(
        feature_dim=config["feature_dim"],
        hidden_dim=config["hidden_dim"],
        num_items=config["num_items"],
        slate_size=config["slate_size"],
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    catalog = Catalog(**payload["catalog"])
    return model, catalog, int(config["seed"])


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, catalog, seed = load_model(args.checkpoint, device)
    _, _, test_loader = create_dataloaders(batch_size=args.batch_size, slate_size=model.slate_size, seed=seed)

    greedy_records = []
    spd_records = []
    lambda_tail = []
    lambda_diversity = []

    for batch in test_loader:
        user_state = batch["user_state"].to(device)
        constraints = batch["constraints"].to(device)

        greedy = decode_greedy(model, user_state, constraints)
        spd = stochastic_primal_dual_decode(model, user_state, constraints, catalog)

        greedy_metrics = slate_metrics(greedy.slates.cpu(), batch["user_state"], batch["constraints"], catalog)
        spd_metrics = slate_metrics(spd.slates.cpu(), batch["user_state"], batch["constraints"], catalog)

        greedy_records.append(greedy_metrics)
        spd_records.append(spd_metrics)
        lambda_tail.append(spd.diagnostics["avg_lambda_tail"])
        lambda_diversity.append(spd.diagnostics["avg_lambda_diversity"])

    def mean_metric(records, key: str) -> float:
        return sum(item[key] for item in records) / max(1, len(records))

    print("== Greedy decode ==")
    print(
        f"relevance={mean_metric(greedy_records, 'relevance'):.4f} business={mean_metric(greedy_records, 'business'):.4f} "
        f"tail_ratio={mean_metric(greedy_records, 'tail_ratio'):.4f} category_coverage={mean_metric(greedy_records, 'category_coverage'):.4f} "
        f"tail_violation={mean_metric(greedy_records, 'tail_violation'):.4f} category_violation={mean_metric(greedy_records, 'category_violation'):.4f}"
    )

    print("== Stochastic primal-dual decode ==")
    print(
        f"relevance={mean_metric(spd_records, 'relevance'):.4f} business={mean_metric(spd_records, 'business'):.4f} "
        f"tail_ratio={mean_metric(spd_records, 'tail_ratio'):.4f} category_coverage={mean_metric(spd_records, 'category_coverage'):.4f} "
        f"tail_violation={mean_metric(spd_records, 'tail_violation'):.4f} category_violation={mean_metric(spd_records, 'category_violation'):.4f}"
    )
    print(
        f"avg_lambda_tail={sum(lambda_tail) / max(1, len(lambda_tail)):.4f} "
        f"avg_lambda_diversity={sum(lambda_diversity) / max(1, len(lambda_diversity)):.4f}"
    )


if __name__ == "__main__":
    main()
