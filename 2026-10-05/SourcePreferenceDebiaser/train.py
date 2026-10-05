import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from dataset import SOURCES, SourcePreferenceDataset, build_counterfactual_batch
from model import SourceAwareRanker, debiasing_loss


def move_batch(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def train_one(model, loader, device, debiased=False, epochs=8, lr=1e-3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for batch in loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad()
            if debiased:
                cf_batch = move_batch(build_counterfactual_batch(batch), device)
                loss, _ = debiasing_loss(model, batch, cf_batch)
            else:
                output = model(batch)
                loss = F.binary_cross_entropy_with_logits(output["pair_logit"], batch["label"])
            loss.backward()
            optimizer.step()
            running += float(loss.detach())
        print(f"epoch={epoch + 1} debiased={debiased} loss={running / max(1, len(loader)):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--output-dir", default="runs/demo")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    dataset = SourcePreferenceDataset(size=2400, seed=args.seed)
    train_set, _ = random_split(dataset, [2000, 400], generator=torch.Generator().manual_seed(args.seed))
    loader = DataLoader(train_set, batch_size=64, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    baseline = SourceAwareRanker(num_sources=len(SOURCES)).to(device)
    debiased = SourceAwareRanker(num_sources=len(SOURCES)).to(device)
    train_one(baseline, loader, device, debiased=False, epochs=args.epochs)
    train_one(debiased, loader, device, debiased=True, epochs=args.epochs)
    torch.save({"model": baseline.state_dict(), "sources": SOURCES}, os.path.join(args.output_dir, "baseline.pt"))
    torch.save({"model": debiased.state_dict(), "sources": SOURCES}, os.path.join(args.output_dir, "debiased.pt"))


if __name__ == "__main__":
    main()
