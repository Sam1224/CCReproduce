import argparse

import torch
from torch.utils.data import DataLoader

from dataset import NeedleDataset
from model import TinyKVModel, kv2_mask, proxy_scores


def accuracy(model, loader, device, mode: str, budget_ratio: float):
    correct = total = 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad():
            hidden, keys, values = model.prefill(batch["tokens"])
            if mode == "full":
                mask = torch.ones(keys.shape[:2], dtype=torch.bool, device=device)
            elif mode == "proxy":
                scores = proxy_scores(batch["tokens"], keys)
                keep = scores.topk(max(1, int(keys.shape[1] * budget_ratio)), dim=1).indices
                mask = torch.zeros(keys.shape[:2], dtype=torch.bool, device=device)
                mask.scatter_(1, keep, True)
            else:
                mask = kv2_mask(batch["tokens"], keys, values, budget_ratio=budget_ratio, refinement_steps=2)
            logits = model.decode(batch["query"], keys, values, mask)
            pred = logits.argmax(dim=-1)
        correct += int((pred == batch["label"]).sum())
        total += int(batch["label"].numel())
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="runs/demo/model.pt")
    parser.add_argument("--budget-ratio", type=float, default=0.2)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.checkpoint, map_location=device)
    model = TinyKVModel().to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    loader = DataLoader(NeedleDataset(size=512, seq_len=payload.get("seq_len", 96), seed=29), batch_size=128)
    print({
        "full_cache_accuracy": accuracy(model, loader, device, "full", args.budget_ratio),
        "proxy_only_accuracy": accuracy(model, loader, device, "proxy", args.budget_ratio),
        "kv2_accuracy": accuracy(model, loader, device, "kv2", args.budget_ratio),
        "budget_ratio": args.budget_ratio,
    })


if __name__ == "__main__":
    main()
