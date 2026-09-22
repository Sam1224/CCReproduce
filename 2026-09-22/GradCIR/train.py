import argparse
import torch
from torch.utils.data import DataLoader, random_split
from dataset import ToyGradCIRDataset
from model import GradCIR, hierarchy_angular_loss, ndcg_at_k


def train(epochs: int = 3, batch_size: int = 32, lr: float = 1e-3, output: str = "gradcir.pt") -> None:
    data = ToyGradCIRDataset()
    train_data, valid_data = random_split(data, [420, len(data) - 420], generator=torch.Generator().manual_seed(42))
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size)
    model = GradCIR()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            scores = model(batch)
            loss = hierarchy_angular_loss(scores, batch["relevance"])
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
        model.eval()
        with torch.no_grad():
            ndcgs = []
            for batch in valid_loader:
                scores = model(batch).diag()
                ndcgs.append(ndcg_at_k(scores, batch["relevance"], k=10))
        print(f"epoch={epoch + 1} loss={total_loss / len(loader):.4f} ndcg@10={sum(ndcgs) / len(ndcgs):.4f}")
    torch.save(model.state_dict(), output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", type=str, default="gradcir.pt")
    args = parser.parse_args()
    train(args.epochs, args.batch_size, args.lr, args.output)
