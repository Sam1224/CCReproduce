import argparse
from pathlib import Path

import torch

from data import OnlineItemPool, make_loaders
from model import TwoTower, TwoTowerConfig, hit_rate_at_k, sampled_softmax_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-neg", type=int, default=31)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--out", type=str, default="checkpoints/goobs.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, catalog, num_users, num_items = make_loaders(args.batch_size)
    pool = OnlineItemPool(catalog.item_cluster)
    model = TwoTower(TwoTowerConfig(num_users, num_items)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for batch in train_loader:
            user = batch["user_id"].to(device)
            pos = batch["item_id"].to(device)
            cluster = batch["cluster_id"].to(device)
            neg = pool.sample_cluster_negatives(pos, cluster, args.num_neg, device)
            logits = model.sampled_logits(user, pos, neg)
            loss = sampled_softmax_loss(logits)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            pool.update(pos.detach().cpu(), cluster.detach().cpu())
            total += loss.item()
        hr50 = hit_rate_at_k(model, test_loader, num_items, 50, device)
        hr100 = hit_rate_at_k(model, test_loader, num_items, 100, device)
        print(f"epoch={epoch} loss={total/len(train_loader):.4f} hr@50={hr50:.4f} hr@100={hr100:.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "num_users": num_users, "num_items": num_items}, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
