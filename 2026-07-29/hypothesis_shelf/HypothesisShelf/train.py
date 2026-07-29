import argparse

import torch
from torch.utils.data import DataLoader

from dataset import ToyShelfDataset, collate_shelves
from model import HypothesisShelfModel, shelf_training_loss


def train(args):
    torch.manual_seed(args.seed)
    dataset = ToyShelfDataset(num_users=args.num_users, seed=args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_shelves)
    model = HypothesisShelfModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    for epoch in range(args.epochs):
        total = 0.0
        for batch in loader:
            outputs = model(batch["profile"], batch["target_type"], batch["catalogue"], batch["catalogue_type"])
            loss = shelf_training_loss(outputs, batch["target_type"], batch["positive_items"])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        print(f"epoch={epoch + 1} loss={total / len(loader):.4f}")
    torch.save(model.state_dict(), args.output)
    print(f"saved={args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-users", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output", type=str, default="hypothesis_shelf.pt")
    train(parser.parse_args())
