from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from data import RankingRowDataset, build_catalog, build_turn_specs, collate_rows
from model import ConstraintAwareRanker


def evaluate(model: ConstraintAwareRanker, loader: DataLoader, loss_fn: nn.Module) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for query_features, candidate_features, labels in loader:
            logits = model(query_features, candidate_features)
            loss = loss_fn(logits, labels)
            preds = (torch.sigmoid(logits) > 0.5).float()
            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return total_loss / max(total, 1), correct / max(total, 1)


def main() -> None:
    torch.manual_seed(42)
    catalog = build_catalog()
    sessions = build_turn_specs()
    dataset = RankingRowDataset(catalog, sessions)

    train_size = int(len(dataset) * 0.85)
    valid_size = len(dataset) - train_size
    train_set, valid_set = random_split(dataset, [train_size, valid_size])

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, collate_fn=collate_rows)
    valid_loader = DataLoader(valid_set, batch_size=128, shuffle=False, collate_fn=collate_rows)

    sample_query, sample_candidate, _ = dataset[0]
    model = ConstraintAwareRanker(sample_query.numel(), sample_candidate.numel())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    best_state = None
    best_valid = float("inf")

    for epoch in range(18):
        model.train()
        for query_features, candidate_features, labels in train_loader:
            optimizer.zero_grad()
            logits = model(query_features, candidate_features)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
        valid_loss, valid_acc = evaluate(model, valid_loader, loss_fn)
        if valid_loss < best_valid:
            best_valid = valid_loss
            best_state = {name: tensor.cpu() for name, tensor in model.state_dict().items()}
        print(f"epoch={epoch + 1:02d} valid_loss={valid_loss:.4f} valid_acc={valid_acc:.4f}")

    out_path = Path(__file__).resolve().parent / "macs_toy.pt"
    torch.save(
        {
            "state_dict": best_state,
            "query_dim": sample_query.numel(),
            "candidate_dim": sample_candidate.numel(),
        },
        out_path,
    )
    print(f"saved checkpoint to {out_path}")


if __name__ == "__main__":
    main()
