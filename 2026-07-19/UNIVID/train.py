import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from data import ToyVideoModerationDataset
from model import UNIVIDLite, loss_fn


def accuracy(logits, labels):
    pred = (torch.sigmoid(logits) >= 0.5).float()
    return (pred == labels).float().mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    ds = ToyVideoModerationDataset(size=768)
    train_ds, val_ds = random_split(ds, [640, 128], generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    model = UNIVIDLite()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for batch in train_loader:
            out = model(batch["frame_features"], batch["text_features"], batch["policy_id"])
            loss = loss_fn(out, batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        model.eval()
        accs = []
        with torch.no_grad():
            for batch in val_loader:
                out = model(batch["frame_features"], batch["text_features"], batch["policy_id"])
                accs.append(accuracy(out["violation_logit"], batch["violation_label"]))
        print(f"epoch={epoch+1} loss={total/len(train_loader):.4f} val_violation_acc={sum(accs)/len(accs):.3f}")

    Path("checkpoints").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/univid_lite.pt")


if __name__ == "__main__":
    main()
