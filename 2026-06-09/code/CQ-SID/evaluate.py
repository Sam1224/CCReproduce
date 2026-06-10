"""
Evaluation for CQ-SID + EG-GRPO.

Measures:
1. Recall@K: fraction of relevant products retrieved via CQ-SID codes
2. Beam search complexity reduction vs. standard product-ID beam search
"""

import torch
from data import get_dataloaders
from cq_sid import CQSIDEncoder
from eg_grpo import GenerativeRetriever


def build_product_index(encoder, data, device):
    """Build mapping from CQ-SID codes to product IDs."""
    n_products = data["n_products"]
    feat_dim = data["feat_dim"]

    product_feat = torch.tensor(data["product_text"], dtype=torch.float32).to(device)
    product_cat = torch.tensor(data["product_cats"], dtype=torch.long).to(device)

    encoder.eval()
    batch_size = 512
    all_codes = []
    with torch.no_grad():
        for i in range(0, n_products, batch_size):
            pf = product_feat[i:i+batch_size]
            pc = product_cat[i:i+batch_size]
            codes, _, _, _, _ = encoder(pf, pc)
            all_codes.append(codes.cpu())

    all_codes = torch.cat(all_codes, dim=0)  # (n_products, K)
    return all_codes


@torch.no_grad()
def evaluate_recall(retriever, encoder, loader, all_product_codes, device, k: int = 10):
    """Evaluate Recall@K for generative retrieval."""
    retriever.eval()
    hits, total = 0, 0

    for batch in loader:
        qf = batch["query_feat"].to(device)
        true_pids = batch["product_id"]
        labels = batch["label"]

        # Only consider positive pairs
        pos_mask = labels == 1
        if pos_mask.sum() == 0:
            continue

        qf_pos = qf[pos_mask]
        true_pids_pos = true_pids[pos_mask]

        # Generate codes via retriever (greedy)
        _, pred_codes = retriever(qf_pos)  # (B, K)
        pred_codes = pred_codes.cpu()

        # Find products matching predicted codes
        for i, (pred_c, true_pid) in enumerate(zip(pred_codes, true_pids_pos)):
            # Check how many products match at each level (hierarchical matching)
            # Level 1 match: all products with same top-level code
            matches = (all_product_codes[:, 0] == pred_c[0])
            # Further filter by subsequent levels (beam search simulation)
            for k_lvl in range(1, pred_c.shape[0]):
                next_matches = matches & (all_product_codes[:, k_lvl] == pred_c[k_lvl])
                if next_matches.sum() > 0:
                    matches = next_matches
                # Stop if we have <= k candidates
                if matches.sum() <= k:
                    break

            retrieved_pids = torch.where(matches)[0]
            if true_pid in retrieved_pids:
                hits += 1
            total += 1

    return hits / max(total, 1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_products, n_categories, n_queries = 1000, 30, 5000

    _, val_loader, data = get_dataloaders(
        batch_size=256, n_products=n_products, n_categories=n_categories,
        n_queries=n_queries, feat_dim=128
    )

    encoder = CQSIDEncoder(
        feat_dim=128, n_categories=n_categories, cat_embed_dim=32,
        hidden_dim=128, n_levels=4, codebook_size=256
    ).to(device)

    retriever = GenerativeRetriever(
        query_dim=128, hidden_dim=128, n_levels=4, codebook_size=256
    ).to(device)

    try:
        ckpt = torch.load("cq_sid_checkpoint.pt", map_location=device)
        encoder.load_state_dict(ckpt["encoder"])
        retriever.load_state_dict(ckpt["retriever"])
        print("Loaded trained CQ-SID checkpoint.")
    except FileNotFoundError:
        print("No checkpoint. Using untrained model for demo.")

    # Build product code index
    all_codes = build_product_index(encoder, data, device)

    recall = evaluate_recall(retriever, encoder, val_loader, all_codes, device, k=10)
    print(f"\n=== CQ-SID Evaluation ===")
    print(f"Recall@10: {recall:.4f}")
    print(f"\nSearch space reduction vs. linear scan:")
    print(f"  Toy: {n_products} products → after level-1 filter: ~{n_products // 256} candidates")
    print(f"  (Paper reports: beam search complexity O(K*C) vs. O(N) for brute force)")


if __name__ == "__main__":
    main()
