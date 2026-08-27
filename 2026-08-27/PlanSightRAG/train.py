from __future__ import annotations

import torch
import torch.nn.functional as F

from data import build_dataloaders
from model import MultiVectorRetriever


def recall_at_5(model, pages, loader) -> float:
    model.eval()
    hit = total = 0
    with torch.no_grad():
        for batch in loader:
            out = model(batch["query_vec"], pages, topk=5)
            hit += int((out.topk_ids == batch["gold_page"].unsqueeze(1)).any(dim=1).sum())
            total += int(batch["gold_page"].numel())
    return hit / max(total, 1)


def main() -> None:
    torch.manual_seed(11)
    pages, train_loader, val_loader, test_loader = build_dataloaders()
    model = MultiVectorRetriever()
    optim = torch.optim.AdamW(model.parameters(), lr=2e-3)
    for epoch in range(5):
        model.train()
        loss_sum = 0.0
        for batch in train_loader:
            out = model(batch["query_vec"], pages, topk=5)
            loss = F.cross_entropy(out.scores, batch["gold_page"])
            optim.zero_grad()
            loss.backward()
            optim.step()
            loss_sum += float(loss.detach())
        print(f"epoch={epoch + 1} loss={loss_sum / len(train_loader):.4f} val_recall@5={recall_at_5(model, pages, val_loader):.3f}")
    torch.save(model.state_dict(), "plansightrag_retriever.pt")
    print(f"test_recall@5={recall_at_5(model, pages, test_loader):.3f}")


if __name__ == "__main__":
    main()
