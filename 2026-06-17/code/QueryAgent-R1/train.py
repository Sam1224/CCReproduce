"""QueryAgent-R1 — Training Script"""

import argparse
import torch
import torch.optim as optim
from tqdm import tqdm

from data import generate_toy_ecommerce_data, build_dataloaders
from model import QueryAgentR1


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = total_ctr = total_rl = 0
    for batch in loader:
        query_seq = batch["query_seq"].to(device)
        next_query = batch["next_query"].to(device)
        retrieved_prods = batch["retrieved_prods"].float().to(device)
        cvr_signal = batch["cvr_signal"].to(device)

        # Map retrieved_prods (product IDs) to embeddings
        prod_embs = model.product_embs[retrieved_prods]  # (B, 5, D)

        out = model(query_seq, next_query, prod_embs, cvr_signal)
        loss = out["loss"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_ctr += out["L_CTR"]
        total_rl += out["L_RL"]

    n = len(loader)
    return total_loss / n, total_ctr / n, total_rl / n


@torch.no_grad()
def evaluate(model, loader, device, top_k=10):
    model.eval()
    hits = total = 0
    for batch in loader:
        query_seq = batch["query_seq"].to(device)
        labels = batch["next_query"].to(device)
        recs = model.recommend_queries(query_seq, top_k=top_k)
        hits += (recs == labels.unsqueeze(-1)).any(dim=-1).sum().item()
        total += labels.shape[0]
    return hits / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lambda_rl", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data = generate_toy_ecommerce_data(seed=args.seed)
    train_loader, val_loader = build_dataloaders(data, batch_size=args.batch_size)

    product_embs = torch.FloatTensor(data["product_embs"])
    model = QueryAgentR1(
        n_queries=data["n_queries"],
        n_products=data["n_products"],
        product_embs=product_embs,
        lambda_rl=args.lambda_rl,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_hit = 0
    print(f"\nTraining QueryAgent-R1 for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        loss, l_ctr, l_rl = train_epoch(model, train_loader, optimizer, device)
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            hit = evaluate(model, val_loader, device)
            if hit > best_hit:
                best_hit = hit
                torch.save(model.state_dict(), "queryagent_r1_best.pt")
            print(
                f"Epoch {epoch:3d} | Loss={loss:.4f} L_CTR={l_ctr:.4f} L_RL={l_rl:.4f} "
                f"| Hit@10={hit:.4f} (best={best_hit:.4f})"
            )

    print(f"\nTraining complete. Best Hit@10: {best_hit:.4f}")


if __name__ == "__main__":
    main()
