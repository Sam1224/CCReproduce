from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from data import SyntheticProductSequenceDataset, collate
from model import DSRec


def main() -> None:
    torch.manual_seed(13)
    dataset = SyntheticProductSequenceDataset(size=384)
    loader = DataLoader(dataset, batch_size=48, shuffle=True, collate_fn=collate)
    model = DSRec()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    for epoch in range(4):
        total = 0.0
        for batch in loader:
            logits = model(batch.items, batch.deltas)
            loss = loss_fn(logits, batch.targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
        history.append(total / len(loader))
        print(f"epoch={epoch + 1} loss={history[-1]:.4f}")
    Path("artifacts").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "artifacts/dsrec.pt")
    Path("artifacts/metrics.json").write_text(json.dumps({"loss": history[-1]}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
