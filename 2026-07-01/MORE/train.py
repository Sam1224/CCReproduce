from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from data import MoreDialogDataset, collate
from model import MorePolicy, metrics


def main():
    torch.manual_seed(0)
    ds = MoreDialogDataset(n=1400, seed=3)
    train_ds, _ = random_split(ds, [1100, 300], generator=torch.Generator().manual_seed(0))
    loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MorePolicy(len(ds.vocab.itos)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    for epoch in range(1, 7):
        total = 0.0
        for batch in loader:
            ids = batch["ids"].to(device)
            reasoning = batch["reasoning"].to(device)
            response = batch["response"].to(device)
            rlogits, ylogits = model(ids)
            loss = model.adaptive_loss(rlogits, ylogits, reasoning, response)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.item())
        print(f"epoch={epoch:02d} loss={total/max(1,len(loader)):.4f} weights={torch.softmax(model.log_reward_weights,0).detach().cpu().tolist()}")
    out = Path(__file__).resolve().parent / "checkpoints"
    out.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "vocab": ds.vocab.itos}, out / "more_policy.pt")


if __name__ == "__main__":
    main()
