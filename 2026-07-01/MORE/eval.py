from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import DialogVocab, MoreDialogDataset, collate
from model import MorePolicy, metrics


@torch.no_grad()
def main():
    root = Path(__file__).resolve().parent
    ckpt = torch.load(root / "checkpoints" / "more_policy.pt", map_location="cpu")
    ds = MoreDialogDataset(n=400, seed=55)
    ds.vocab = DialogVocab()
    ds.vocab.itos = ckpt["vocab"]
    ds.vocab.stoi = {t: i for i, t in enumerate(ds.vocab.itos)}
    model = MorePolicy(len(ds.vocab.itos))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    loader = DataLoader(ds, batch_size=128, shuffle=False, collate_fn=collate)
    ra = rq = both = n = 0.0
    for batch in loader:
        rlogits, ylogits = model(batch["ids"])
        a, q, b = metrics(rlogits, ylogits, batch["reasoning"], batch["response"])
        bs = len(batch["response"])
        ra += a * bs; rq += q * bs; both += b * bs; n += bs
    print(f"reasoning_acc={ra/n:.3f} response_quality_acc={rq/n:.3f} joint_success={both/n:.3f}")


if __name__ == "__main__":
    main()
