import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import HiddenMessageDataset, ToyConfig, build_template_bank
from model import AdaptiveViewRetrieval, avr_loss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", type=str, default="avr_toy.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def run_epoch(model, loader, templates, optimizer=None, device="cpu"):
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "retrieval_acc": 0.0, "moderation_acc": 0.0}
    count = 0
    for batch in loader:
        images = batch["image"].to(device)
        message_id = batch["message_id"].to(device)
        harmful = batch["harmful"].to(device)
        outputs = model(images, templates)
        loss, metrics = avr_loss(outputs, message_id, harmful)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        batch_size = images.size(0)
        for key in totals:
            totals[key] += metrics[key] * batch_size
        count += batch_size
    return {key: value / max(count, 1) for key, value in totals.items()}


def main():
    args = parse_args()
    config = ToyConfig()
    train_set = HiddenMessageDataset("train", config)
    val_set = HiddenMessageDataset("val", config)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)
    device = torch.device(args.device)
    model = AdaptiveViewRetrieval().to(device)
    templates = build_template_bank(config).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    best_val = -1.0
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, templates, optimizer, device)
        val_metrics = run_epoch(model, val_loader, templates, None, device)
        score = 0.5 * (val_metrics["retrieval_acc"] + val_metrics["moderation_acc"])
        print(f"epoch={epoch} train={train_metrics} val={val_metrics}")
        if score > best_val:
            best_val = score
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "config": config.__dict__}, args.output)
    print(f"saved best checkpoint to {args.output} with validation score {best_val:.4f}")


if __name__ == "__main__":
    main()
