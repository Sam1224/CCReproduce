import argparse
from pathlib import Path

import torch

from data import make_loaders
from model import GMOE2DITToy, loss_fn, mask_iou


def evaluate(model, loader, device):
    model.eval()
    op_ok = 0
    count = 0
    src_iou = 0.0
    tgt_iou = 0.0
    l1 = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            op_ok += (out["op_logits"].argmax(dim=-1) == batch["op_id"]).sum().item()
            count += batch["op_id"].numel()
            src_iou += mask_iou(out["source_logits"], batch["source_mask"])
            tgt_iou += mask_iou(out["target_logits"], batch["target_mask"])
            l1 += torch.nn.functional.l1_loss(out["edited"], batch["target"]).item()
    n_batches = max(len(loader), 1)
    return {"op_acc": op_ok / max(count, 1), "source_iou": src_iou / n_batches, "target_iou": tgt_iou / n_batches, "edit_l1": l1 / n_batches}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--out", type=str, default="checkpoints/gmo_e2dit.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = make_loaders(args.batch_size)
    model = GMOE2DITToy().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch), batch)
            loss.backward()
            opt.step()
            total += loss.item()
        metrics = evaluate(model, test_loader, device)
        print(f"epoch={epoch} loss={total/len(train_loader):.4f} metrics={metrics}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "metrics": evaluate(model, test_loader, device)}, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
