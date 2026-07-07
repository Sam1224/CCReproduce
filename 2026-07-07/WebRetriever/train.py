import argparse
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from dataset import SyntheticWebRetrieverDataset, WebTaskConfig
from model import WebRetrieverAgent, webretriever_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    config = WebTaskConfig(samples=args.samples)
    dataset = SyntheticWebRetrieverDataset(config)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = WebRetrieverAgent(
        feature_dim=config.feature_dim,
        action_steps=config.action_steps,
        action_vocab=config.action_vocab,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for epoch in range(args.epochs):
        metrics = defaultdict(float)
        batches = 0
        for batch in loader:
            batch = {name: tensor.to(device) for name, tensor in batch.items()}
            outputs = model(batch["query"], batch["pages"])
            losses = webretriever_loss(outputs, batch)
            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()
            for name, value in losses.items():
                metrics[name] += float(value.detach().cpu())
            batches += 1
        summary = " ".join(f"{name}={value / batches:.4f}" for name, value in sorted(metrics.items()))
        print(f"epoch={epoch + 1} {summary}")

    torch.save({"model": model.state_dict(), "config": config.__dict__}, "webretriever_toy.pt")
    print("saved webretriever_toy.pt")


if __name__ == "__main__":
    main()
