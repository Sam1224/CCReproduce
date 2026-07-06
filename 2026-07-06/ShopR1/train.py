import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import ShoppingBehaviorDataset, collate_batch, save_toy_dataset
from model import ShopR1Policy


def move_batch(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def train(args):
    if args.write_toy_data:
        save_toy_dataset(args.data)
    dataset = ShoppingBehaviorDataset(args.data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    device = torch.device(args.device)
    model = ShopR1Policy(vocab_size=len(dataset.vocab.token_to_id)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad()
            sft_loss = model.supervised_loss(batch)
            rl_loss = model.grpo_style_loss(batch)
            loss = sft_loss + args.rl_weight * rl_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        print(f"epoch={epoch + 1} loss={total_loss / max(len(loader), 1):.4f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "vocab": dataset.vocab.token_to_id}, output_dir / "shop_r1.pt")
    print(f"saved checkpoint to {output_dir / 'shop_r1.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="toy_shopping.json")
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--rl-weight", type=float, default=0.2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--write-toy-data", action="store_true")
    train(parser.parse_args())
