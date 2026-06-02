"""
Evaluation script for TGQ-Former.
Computes Hit Rate @ K (H@K) for I2I retrieval.

Paper: TGQ-Former consistently outperforms strong baselines,
improving H@100 by 6.04% on average on large-scale e-commerce datasets.
"""
import argparse
import torch
import torch.nn.functional as F
import numpy as np

from model import TGQFormer
from dataset import get_dataloaders, ToyECommerceDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--k_values", type=str, default="1,5,10,50,100")
    return p.parse_args()


@torch.no_grad()
def compute_embeddings(model, dataset, batch_size, device):
    """Compute embeddings for all items in the dataset."""
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=batch_size)
    all_embs = []
    all_cats = []
    model.eval()
    for batch in loader:
        visual = batch["visual_tokens"].to(device)
        meta_ids = batch["meta_ids"].to(device)
        meta_mask = batch["meta_mask"].to(device)
        emb = model.encode_item(visual, meta_ids, meta_mask)
        all_embs.append(emb.cpu())
        all_cats.extend(batch["category"].tolist())
    return torch.cat(all_embs, dim=0), torch.tensor(all_cats)


def hit_rate_at_k(query_embs, gallery_embs, query_cats, gallery_cats, k: int) -> float:
    """Compute Hit Rate @ K: fraction of queries with ≥1 same-category item in top-K."""
    sim = torch.mm(query_embs, gallery_embs.t())  # (N, M)
    # Mask self (assume gallery == query set)
    sim.fill_diagonal_(-1e9)

    top_k_idx = sim.topk(k, dim=-1).indices  # (N, k)
    retrieved_cats = gallery_cats[top_k_idx]  # (N, k)
    query_cats_expanded = query_cats.unsqueeze(1).expand_as(retrieved_cats)
    hits = (retrieved_cats == query_cats_expanded).any(dim=-1)  # (N,)
    return hits.float().mean().item()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader, test_base = get_dataloaders(args.batch_size)

    model = TGQFormer(visual_dim=2048, hidden_dim=256, num_queries=16, embed_dim=128).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    print("Computing embeddings...")
    embs, cats = compute_embeddings(model, test_base, args.batch_size, device)

    k_values = [int(k) for k in args.k_values.split(",")]
    print("\n=== TGQ-Former Evaluation ===")
    print(f"Gallery size: {len(embs)}")
    for k in k_values:
        hr = hit_rate_at_k(embs, embs, cats, cats, k)
        print(f"  H@{k:4d}: {hr:.4f}")

    print("\n(Paper: +6.04% avg H@100 over strong connector baselines on JD.COM data)")


if __name__ == "__main__":
    main()
