import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from data import ToyAVEDataset
from model import AttributeValueExtractor, LLMArena, cleaning_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    ds = ToyAVEDataset(size=1400)
    train_ds, val_ds = random_split(ds, [1100, 300], generator=torch.Generator().manual_seed(0))
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    model = AttributeValueExtractor()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4)

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for batch in loader:
            logits = model(batch["features"])
            loss = torch.nn.functional.cross_entropy(logits, batch["value_id"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        model.eval()
        correct = count = 0
        with torch.no_grad():
            for batch in val_loader:
                pred = model(batch["features"]).argmax(dim=-1)
                correct += (pred == batch["value_id"]).sum().item()
                count += pred.numel()
        print(f"epoch={epoch+1} loss={total/len(loader):.4f} val_attr_acc={correct/count:.3f}")

    Path("checkpoints").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/synthave_extractor.pt")

    arena = LLMArena()
    batch = next(iter(val_loader))
    majority, agreement, _ = arena.vote(batch["features"], batch["synthetic_label"])
    print(cleaning_metrics(majority, agreement, batch["value_id"], batch["synthetic_label"]))


if __name__ == "__main__":
    main()
