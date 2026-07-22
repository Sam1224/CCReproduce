import argparse

import torch
from torch.utils.data import DataLoader

from dataset import HiddenMessageDataset, ToyConfig, build_template_bank
from model import AdaptiveViewRetrieval


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="avr_toy.pt")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def balanced_accuracy(pred, target):
    pred = pred.bool()
    target = target.bool()
    positives = target.sum().clamp_min(1)
    negatives = (~target).sum().clamp_min(1)
    tpr = (pred & target).sum().float() / positives
    tnr = ((~pred) & (~target)).sum().float() / negatives
    return 0.5 * (tpr + tnr)


def main():
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = ToyConfig(**checkpoint.get("config", {}))
    model = AdaptiveViewRetrieval().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    templates = build_template_bank(config).to(device)
    loader = DataLoader(HiddenMessageDataset("test", config), batch_size=args.batch_size)
    all_message, all_message_pred, all_harmful, all_harmful_pred = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch["image"].to(device), templates)
            all_message.append(batch["message_id"])
            all_message_pred.append(outputs["retrieval_logits"].argmax(dim=-1).cpu())
            all_harmful.append(batch["harmful"])
            all_harmful_pred.append((outputs["harmful_logit"].sigmoid().cpu() >= 0.5).float())
    message = torch.cat(all_message)
    message_pred = torch.cat(all_message_pred)
    harmful = torch.cat(all_harmful)
    harmful_pred = torch.cat(all_harmful_pred)
    retrieval_acc = (message_pred == message).float().mean().item()
    bal_acc = balanced_accuracy(harmful_pred, harmful).item()
    print({"retrieval_accuracy": round(retrieval_acc, 4), "balanced_accuracy": round(bal_acc, 4)})


if __name__ == "__main__":
    main()
