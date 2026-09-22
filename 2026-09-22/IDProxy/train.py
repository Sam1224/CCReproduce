import argparse
import torch
from torch.utils.data import DataLoader, random_split
from dataset import ToyIDProxyDataset
from model import IDProxyCTR, idproxy_loss


def auc_score(logits: torch.Tensor, labels: torch.Tensor) -> float:
    order = torch.argsort(logits)
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(1, logits.numel() + 1, device=logits.device).float()
    positives = labels.bool()
    num_pos = positives.sum().float()
    num_neg = labels.numel() - num_pos
    if num_pos == 0 or num_neg == 0:
        return 0.5
    auc = (ranks[positives].sum() - num_pos * (num_pos + 1) / 2) / (num_pos * num_neg)
    return float(auc.cpu())


def train(epochs: int = 3, batch_size: int = 64, lr: float = 1e-3, output: str = "idproxy.pt") -> None:
    dataset = ToyIDProxyDataset()
    train_data, valid_data = random_split(dataset, [1700, len(dataset) - 1700], generator=torch.Generator().manual_seed(42))
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size)
    model = IDProxyCTR()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for batch in loader:
            optimizer.zero_grad()
            output_dict = model(batch)
            loss = idproxy_loss(output_dict, batch["label"], batch["is_cold"])
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
        model.eval()
        logits, labels = [], []
        with torch.no_grad():
            for batch in valid_loader:
                logits.append(model(batch)["logits"])
                labels.append(batch["label"])
        print(f"epoch={epoch + 1} loss={total / len(loader):.4f} auc={auc_score(torch.cat(logits), torch.cat(labels)):.4f}")
    torch.save(model.state_dict(), output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", type=str, default="idproxy.pt")
    args = parser.parse_args()
    train(args.epochs, args.batch_size, args.lr, args.output)
