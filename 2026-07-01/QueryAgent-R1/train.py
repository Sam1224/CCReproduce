from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from data import QueryAgentDataset, collate
from model import QueryAgentR1


def train_epoch(model, loader, opt, device):
    model.train()
    total = 0.0
    for batch in loader:
        hist = batch["history"].to(device)
        cand = batch["candidates"].to(device)
        labels = batch["labels"].to(device)
        logits = model(hist, cand)
        ce = torch.nn.functional.cross_entropy(logits, labels)
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(-1).mean()
        loss = ce - 0.01 * entropy
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        total += float(loss.item())
    return total / max(1, len(loader))


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = QueryAgentDataset(n=1400, seed=1)
    train_ds, val_ds = random_split(ds, [1100, 300], generator=torch.Generator().manual_seed(0))
    loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=lambda b: collate(b, ds.vocab))
    model = QueryAgentR1(len(ds.vocab.itos)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    for epoch in range(1, 7):
        loss = train_epoch(model, loader, opt, device)
        print(f"epoch={epoch:02d} loss={loss:.4f}")
    out = Path(__file__).resolve().parent / "checkpoints"
    out.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "vocab": ds.vocab.itos}, out / "queryagent_r1.pt")
    print(f"saved {out / 'queryagent_r1.pt'}")


if __name__ == "__main__":
    main()
