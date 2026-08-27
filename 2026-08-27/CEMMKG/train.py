from __future__ import annotations

import torch
import torch.nn.functional as F

from data import VOCAB, build_dataloaders
from model import CEMMKGModel, context_alignment_loss


def evaluate(model: CEMMKGModel, loader) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            out = model(batch)
            pred = out.logits.argmax(dim=-1)
            correct += int((pred == batch["label"]).sum())
            total += int(batch["label"].numel())
    return correct / max(total, 1)


def main() -> None:
    torch.manual_seed(7)
    train_loader, val_loader, test_loader = build_dataloaders()
    model = CEMMKGModel(vocab_size=len(VOCAB))
    optim = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    for epoch in range(4):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            out = model(batch)
            ce = F.cross_entropy(out.logits, batch["label"])
            loss = ce + 0.05 * context_alignment_loss(out)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += float(loss.detach())
        print(f"epoch={epoch + 1} loss={total_loss / len(train_loader):.4f} val_acc={evaluate(model, val_loader):.3f}")
    torch.save(model.state_dict(), "cemmkg_toy.pt")
    print(f"test_acc={evaluate(model, test_loader):.3f}")


if __name__ == "__main__":
    main()
