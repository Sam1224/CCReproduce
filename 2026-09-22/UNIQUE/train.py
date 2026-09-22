import argparse
import torch
from torch.utils.data import DataLoader, random_split
from dataset import ToyFeedDataset
from model import UNIQUE, unique_loss


def train(epochs: int = 3, batch_size: int = 64, lr: float = 1e-3, output: str = "unique.pt") -> None:
    dataset = ToyFeedDataset()
    train_data, valid_data = random_split(dataset, [1700, len(dataset) - 1700], generator=torch.Generator().manual_seed(42))
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size)
    model = UNIQUE()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for batch in loader:
            optimizer.zero_grad()
            output_dict = model(batch)
            loss = unique_loss(output_dict, batch)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
        model.eval()
        ctr_losses = []
        with torch.no_grad():
            for batch in valid_loader:
                output_dict = model(batch)
                ctr_losses.append(torch.nn.functional.binary_cross_entropy_with_logits(output_dict["ctr"], batch["click"].float()).item())
        print(f"epoch={epoch + 1} loss={total / len(loader):.4f} valid_ctr_bce={sum(ctr_losses) / len(ctr_losses):.4f}")
    torch.save(model.state_dict(), output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", type=str, default="unique.pt")
    args = parser.parse_args()
    train(args.epochs, args.batch_size, args.lr, args.output)
