import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from data import ToyEGEODataset
from model import GEOReranker, PromptMetaOptimizer, listwise_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    ds = ToyEGEODataset(queries=512)
    train_ds, val_ds = random_split(ds, [420, 92], generator=torch.Generator().manual_seed(0))
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    ranker = GEOReranker()
    opt = torch.optim.AdamW(ranker.parameters(), lr=5e-4)

    for epoch in range(args.epochs):
        ranker.train()
        total = 0.0
        for batch in loader:
            scores = ranker(batch["query_features"], batch["listing_features"])
            loss = listwise_loss(scores, batch["labels"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        ranker.eval()
        top1 = []
        with torch.no_grad():
            for batch in val_loader:
                scores = ranker(batch["query_features"], batch["listing_features"])
                top1.append((scores.argmax(dim=-1) == batch["labels"].argmax(dim=-1)).float().mean().item())
        print(f"epoch={epoch+1} listwise_loss={total/len(loader):.4f} val_top1={sum(top1)/len(top1):.3f}")

    meta = PromptMetaOptimizer()
    meta_opt = torch.optim.AdamW(meta.parameters(), lr=0.2)
    for _ in range(50):
        w = meta.weights()
        reward = torch.tensor([0.25, 0.30, 0.20, 0.22, 0.0])
        manipulation_penalty = torch.tensor([0.02, 0.01, 0.03, 0.02, 0.0])
        loss = -(w * (reward - manipulation_penalty)).sum()
        meta_opt.zero_grad()
        loss.backward()
        meta_opt.step()

    Path("checkpoints").mkdir(exist_ok=True)
    torch.save({"ranker": ranker.state_dict(), "meta_logits": meta.logits.detach()}, "checkpoints/egeo_ranker.pt")
    print({"best_prompt_family": meta.choose(), "prompt_weights": [round(float(x), 3) for x in meta.weights().detach()]})


if __name__ == "__main__":
    main()
