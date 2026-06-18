"""
AIR (Atomic Intent Reasoning) — Training Script
"""

import argparse
import torch
import torch.optim as optim
from tqdm import tqdm

from data import generate_toy_data, assign_atomic_intents, build_dataloaders
from model import AIRRankingModel


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = total_rank = total_sc = 0
    for batch in loader:
        content_seq = batch["content_seq"].to(device)
        commerce_seq = batch["commerce_seq"].to(device)
        label = batch["label"].to(device)

        out = model(content_seq, commerce_seq, label)
        loss = out["loss"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_rank += out["L_rank"]
        total_sc += out["L_sc"]

    n = len(loader)
    return total_loss / n, total_rank / n, total_sc / n


@torch.no_grad()
def evaluate(model, loader, device, all_item_embs, top_k=10):
    model.eval()
    hits = 0
    total = 0
    for batch in loader:
        content_seq = batch["content_seq"].to(device)
        commerce_seq = batch["commerce_seq"].to(device)
        labels = batch["label"].to(device)

        rec_ids = model.recommend(content_seq, commerce_seq, all_item_embs, top_k=top_k)
        # Hit@K
        hits += (rec_ids == labels.unsqueeze(-1)).any(dim=-1).sum().item()
        total += labels.shape[0]

    return hits / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--intent_dim", type=int, default=128)
    parser.add_argument("--top_k_atoms", type=int, default=8)
    parser.add_argument("--sc_lambda", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    print("Generating data...")
    data = generate_toy_data(seed=args.seed)
    train_loader, val_loader = build_dataloaders(data, batch_size=args.batch_size)

    # Offline atomic intent prototypes (simulated LLM output)
    atom_proto = torch.FloatTensor(data["atom_prototypes"])

    # Model
    model = AIRRankingModel(
        n_content_items=data["n_content_items"],
        n_commerce_items=data["n_commerce_items"],
        atom_prototypes=atom_proto,
        embed_dim=args.embed_dim,
        intent_dim=args.intent_dim,
        top_k_atoms=args.top_k_atoms,
        sc_lambda=args.sc_lambda,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # All item embeddings for recommendation evaluation
    all_item_embs = model.item_encoder.item_emb.weight.detach().to(device)

    print(f"\nTraining AIR for {args.epochs} epochs...")
    best_hit = 0
    for epoch in range(1, args.epochs + 1):
        loss, l_rank, l_sc = train_epoch(model, train_loader, optimizer, device)
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            # Refresh item embs (they change during training)
            all_item_embs = model.item_encoder.item_emb.weight.detach().to(device)
            hit_at_10 = evaluate(model, val_loader, device, all_item_embs, top_k=10)
            if hit_at_10 > best_hit:
                best_hit = hit_at_10
                torch.save(model.state_dict(), "air_best.pt")
            print(
                f"Epoch {epoch:3d} | Loss={loss:.4f} L_rank={l_rank:.4f} L_sc={l_sc:.4f} "
                f"| Hit@10={hit_at_10:.4f} (best={best_hit:.4f})"
            )

    print(f"\nTraining complete. Best Hit@10: {best_hit:.4f}")


if __name__ == "__main__":
    main()
