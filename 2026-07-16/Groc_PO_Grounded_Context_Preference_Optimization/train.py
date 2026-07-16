import argparse
from pathlib import Path

import torch

from data import make_loaders
from model import GroundedContextEncoder, GrocPOConfig, groc_po_loss, preference_accuracy


def evaluate(model, loader, device):
    model.eval()
    total_acc = 0.0
    total_margin = 0.0
    counts = 0
    stage_hits = torch.zeros(3)
    stage_counts = torch.zeros(3)
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            total_acc += preference_accuracy(out, batch)
            total_margin += out["margin"].abs().mean().item()
            counts += 1
            pred = (out["margin"] > 0).float()
            ok = (pred == batch["label"]).float().cpu()
            for sid in range(3):
                mask = (batch["stage"].cpu() == sid)
                stage_hits[sid] += ok[mask].sum()
                stage_counts[sid] += mask.sum()
    return {
        "pref_acc": total_acc / max(counts, 1),
        "abs_margin": total_margin / max(counts, 1),
        "stage_acc": (stage_hits / stage_counts.clamp_min(1)).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--out", type=str, default="checkpoints/groc_po.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = make_loaders(args.batch_size)
    cfg = GrocPOConfig()
    model = GroundedContextEncoder(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            out = model(batch)
            loss = groc_po_loss(out, batch, beta=cfg.beta)
            loss.backward()
            opt.step()
            total += loss.item()
        metrics = evaluate(model, test_loader, device)
        print(f"epoch={epoch} loss={total/len(train_loader):.4f} metrics={metrics}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "metrics": evaluate(model, test_loader, device), "cfg": cfg.__dict__}, out_path)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
