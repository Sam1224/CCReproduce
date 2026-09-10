import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import collate_fn, split_dataset
from model import ContrastiveMetadataTagger


def evaluate(model, loader, device):
    model.eval()
    true_positive = false_positive = false_negative = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(batch["input_ids"])
            predictions = (torch.sigmoid(outputs["logits"]) > 0.5).float()
            labels = batch["labels"]
            true_positive += (predictions * labels).sum().item()
            false_positive += (predictions * (1 - labels)).sum().item()
            false_negative += ((1 - predictions) * labels).sum().item()
    precision = true_positive / max(true_positive + false_positive, 1.0)
    recall = true_positive / max(true_positive + false_negative, 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    f2 = 5 * precision * recall / max(4 * precision + recall, 1e-8)
    return {"precision": precision, "recall": recall, "f1": f1, "f2": f2}


def train(args):
    torch.manual_seed(args.seed)
    train_set, valid_set, tokenizer = split_dataset(samples_per_tag=args.samples_per_tag, seed=args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = ContrastiveMetadataTagger(vocab_size=tokenizer.vocab_size, dim=args.dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model.loss(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        metrics = evaluate(model, valid_loader, device)
        print(f"epoch={epoch + 1} loss={total_loss / max(len(train_loader), 1):.4f} f1={metrics['f1']:.3f} f2={metrics['f2']:.3f}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "vocab_size": tokenizer.vocab_size, "dim": args.dim}, output_dir / "glyph_toy.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a compact Glyph-style metadata tagger")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--samples-per-tag", type=int, default=80)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output-dir", default="artifacts")
    train(parser.parse_args())
