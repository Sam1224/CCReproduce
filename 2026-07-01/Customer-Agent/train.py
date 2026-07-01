from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from data import QuestionVocab, ShopTrajQADataset, collate
from model import CustomerAgent, sft_rlvr_loss


def main():
    torch.manual_seed(0)
    root = Path(__file__).resolve().parent
    ds = ShopTrajQADataset(root / "toy_trajectories", n=96)
    vocab = QuestionVocab([e.question for e in ds.examples])
    train_ds, _ = random_split(ds, [230, len(ds) - 230], generator=torch.Generator().manual_seed(0))
    loader = DataLoader(train_ds, batch_size=48, shuffle=True, collate_fn=lambda b: collate(b, vocab))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CustomerAgent(len(vocab.itos)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    for epoch in range(1, 8):
        total = 0.0
        for batch in loader:
            tmpl, ans = model(batch["question_ids"].to(device))
            loss = sft_rlvr_loss(tmpl, ans, batch["template"].to(device), batch["answer"].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.item())
        print(f"epoch={epoch:02d} loss={total/max(1,len(loader)):.4f}")
    out = root / "checkpoints"
    out.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "vocab": vocab.itos}, out / "customer_agent.pt")


if __name__ == "__main__":
    main()
