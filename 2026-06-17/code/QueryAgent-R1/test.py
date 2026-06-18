"""QueryAgent-R1 — Evaluation Script"""

import argparse
import torch
from data import generate_toy_ecommerce_data, build_dataloaders
from model import QueryAgentR1


def recall_ndcg(recs, labels, k):
    hit = (recs[:, :k] == labels.unsqueeze(-1)).any(dim=-1).float()
    positions = torch.arange(1, k + 1, device=recs.device).float()
    dcg = ((recs[:, :k] == labels.unsqueeze(-1)).float() / torch.log2(positions + 1)).sum(-1)
    return hit.mean().item(), dcg.mean().item()


@torch.no_grad()
def full_eval(model, loader, device, ks=(10, 20)):
    model.eval()
    all_recs, all_labels = [], []
    for batch in loader:
        recs = model.recommend_queries(batch["query_seq"].to(device), top_k=max(ks))
        all_recs.append(recs.cpu())
        all_labels.append(batch["next_query"])
    all_recs = torch.cat(all_recs)
    all_labels = torch.cat(all_labels)
    results = {}
    for k in ks:
        r, n = recall_ndcg(all_recs, all_labels, k)
        results[f"Recall@{k}"] = r
        results[f"NDCG@{k}"] = n
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="queryagent_r1_best.pt")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = generate_toy_ecommerce_data()
    _, val_loader = build_dataloaders(data, batch_size=args.batch_size)

    model = QueryAgentR1(
        n_queries=data["n_queries"],
        n_products=data["n_products"],
        product_embs=torch.FloatTensor(data["product_embs"]),
    ).to(device)

    try:
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        print(f"Loaded {args.ckpt}")
    except FileNotFoundError:
        print("No checkpoint; evaluating random init.")

    results = full_eval(model, val_loader, device)
    print("\n=== QueryAgent-R1 Evaluation ===")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
