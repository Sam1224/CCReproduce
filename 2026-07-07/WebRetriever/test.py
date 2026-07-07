import argparse

import torch
from torch.utils.data import DataLoader

from dataset import SyntheticWebRetrieverDataset, WebTaskConfig
from model import WebRetrieverAgent, evaluate_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--checkpoint", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    config = WebTaskConfig(samples=args.samples, seed=11)
    dataset = SyntheticWebRetrieverDataset(config)
    loader = DataLoader(dataset, batch_size=args.batch_size)
    model = WebRetrieverAgent(
        feature_dim=config.feature_dim,
        action_steps=config.action_steps,
        action_vocab=config.action_vocab,
    ).to(device)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model"])
    model.eval()

    totals = {"page_arrival": 0.0, "action_match": 0.0, "completion_accuracy": 0.0}
    batches = 0
    with torch.no_grad():
        for batch in loader:
            batch = {name: tensor.to(device) for name, tensor in batch.items()}
            outputs = model(batch["query"], batch["pages"])
            metrics = evaluate_outputs(outputs, batch)
            for name, value in metrics.items():
                totals[name] += value
            batches += 1

    for name, value in totals.items():
        print(f"{name}: {value / batches:.4f}")


if __name__ == "__main__":
    main()
