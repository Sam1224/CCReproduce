from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import QueryAgentDataset, Vocab, collate
from model import QueryAgentR1


@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(__file__).resolve().parent
    ckpt = torch.load(root / "checkpoints" / "queryagent_r1.pt", map_location="cpu")
    ds = QueryAgentDataset(n=400, seed=99)
    ds.vocab = Vocab([])
    ds.vocab.itos = ckpt["vocab"]
    ds.vocab.stoi = {t: i for i, t in enumerate(ds.vocab.itos)}
    model = QueryAgentR1(len(ds.vocab.itos)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=lambda b: collate(b, ds.vocab))
    q_em = 0
    item_hit = 0
    cons = 0
    n = 0
    for batch in loader:
        logits = model(batch["history"].to(device), batch["candidates"].to(device)).cpu()
        pred = logits.argmax(-1)
        labels = batch["labels"]
        q_em += int((pred == labels).sum())
        for i, pidx in enumerate(pred.tolist()):
            q = ds.vocab.itos[0] if False else ds.samples[n + i]["candidates"][pidx]
            toks = set(q.split())
            retrieved = [p.pid for p in ds.products if p.category in toks or p.intent in toks][:10]
            hit = any(pid in batch["target_sets"][i] for pid in retrieved)
            item_hit += int(hit)
            cons += int(hit and pidx == int(labels[i]))
        n += len(labels)
    print(f"Q_EM={q_em/n:.3f} I_Hit@10={item_hit/n:.3f} Cons@1={cons/n:.3f}")


if __name__ == "__main__":
    main()
