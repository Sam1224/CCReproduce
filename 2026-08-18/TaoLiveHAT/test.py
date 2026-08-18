import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import benchmark_cases, make_splits
from model import TaoLiveHATModel


def evaluate(model, dataset, device):
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            logits = model(x).logits
            preds = logits.argmax(dim=-1)
            total += y.numel()
            correct += (preds.cpu() == y).sum().item()
    return correct / max(total, 1)


def latency_seconds(model, device):
    sample = torch.randn(1, 24, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(300):
            _ = model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) / 300.0


def load_model(path: Path, device: torch.device):
    model = TaoLiveHATModel().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts")
    args = parser.parse_args()

    artifacts = Path(args.artifacts)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = make_splits()

    fixed = load_model(artifacts / "fixed_harness.pt", device)
    hat = load_model(artifacts / "hat.pt", device)

    report = {
        "benchmarks": benchmark_cases(),
        "fixed": {
            "live_stream_qa": round(evaluate(fixed, splits["base_test"], device), 4),
            "harness_variant_qa": round(evaluate(fixed, splits["variant_test"], device), 4),
        },
        "hat": {
            "live_stream_qa": round(evaluate(hat, splits["base_test"], device), 4),
            "harness_variant_qa": round(evaluate(hat, splits["variant_test"], device), 4),
            "avg_latency_seconds": round(latency_seconds(hat, device), 6),
        },
    }

    with open(artifacts / "test_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
