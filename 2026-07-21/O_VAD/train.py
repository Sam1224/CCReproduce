import argparse
from pathlib import Path

import torch

from data import make_loaders
from model import OvadToy, frame_iou, loss_fn


def evaluate(model, loader, device):
    model.eval()
    video_ok = 0
    object_ok = 0
    count = 0
    state_ok = 0
    state_total = 0
    iou = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            video_ok += ((torch.sigmoid(out["video_logits"]) > 0.5).float() == batch["video_label"]).sum().item()
            object_pred = out["object_logits"].argmax(dim=-1)
            valid = batch["anomaly_object"] >= 0
            if valid.any():
                object_ok += (object_pred[valid] == batch["anomaly_object"][valid]).sum().item()
                count += valid.sum().item()
            iou += frame_iou(out["frame_logits"], batch["frame_mask"])
            state_pred = out["state_logits"].argmax(dim=-1)
            state_ok += (state_pred == batch["states"]).sum().item()
            state_total += batch["states"].numel()
    return {
        "video_accuracy": video_ok / max(len(loader.dataset), 1),
        "object_accuracy": object_ok / max(count, 1),
        "frame_iou": iou / max(len(loader), 1),
        "state_accuracy": state_ok / max(state_total, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--out", type=str, default="checkpoints/o_vad.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = make_loaders(args.batch_size)
    model = OvadToy().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch), batch)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"epoch={epoch} loss={total_loss/len(train_loader):.4f} metrics={evaluate(model, test_loader, device)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "metrics": evaluate(model, test_loader, device)}, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
