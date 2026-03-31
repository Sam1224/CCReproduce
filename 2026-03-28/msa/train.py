from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from dataset import MemoryQADataset, batch_iter
from model import MSAConfig, MSAModel


def eval_acc(model: MSAModel, ds: MemoryQADataset, *, n: int = 200) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(min(n, len(ds))):
            s = ds[i]
            docs = torch.from_numpy(s.docs).unsqueeze(0)
            query = torch.from_numpy(s.query).unsqueeze(0)
            logits, _ = model(docs, query)
            pred = int(logits.argmax(dim=-1).item())
            correct += int(pred == s.answer)
    model.train()
    return correct / max(1, min(n, len(ds)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    cfg = MSAConfig()
    train_ds = MemoryQADataset(n_samples=8000, n_docs=cfg.n_docs, doc_len=cfg.doc_len, vocab_size=cfg.vocab_size, seed=0)
    val_ds = MemoryQADataset(n_samples=1000, n_docs=cfg.n_docs, doc_len=cfg.doc_len, vocab_size=cfg.vocab_size, seed=1)

    model = MSAModel(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)

    it = batch_iter(train_ds, batch_size=args.batch, seed=0)
    for step in range(1, args.steps + 1):
        b = next(it)
        logits, _ = model(b.docs, b.query)
        loss = F.cross_entropy(logits, b.answer)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 100 == 0:
            acc = eval_acc(model, val_ds, n=300)
            print(f"step={step} loss={loss.item():.4f} val_acc={acc:.3f}")

    acc = eval_acc(model, val_ds, n=500)
    print(f"done: val_acc={acc:.3f}")


if __name__ == "__main__":
    main()
