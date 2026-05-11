"""
AFMRL Evaluation Script
arXiv: 2604.20135

Evaluates fine-grained product retrieval performance:
  - Recall@K (standard retrieval metric)
  - MRR (Mean Reciprocal Rank)
  - Attribute quality (attribute prediction accuracy as proxy)
"""

import torch
import numpy as np
from model import AFMRL
from train import generate_ecommerce_toy_data


def mrr(query_embs: torch.Tensor, pos_embs: torch.Tensor) -> float:
    """Mean Reciprocal Rank."""
    sims = query_embs @ pos_embs.T
    ranks = (sims >= sims.diag().unsqueeze(-1)).sum(dim=-1).float()
    return (1.0 / ranks).mean().item()


def recall_at_k(query_embs: torch.Tensor, pos_embs: torch.Tensor, k: int) -> float:
    sims = query_embs @ pos_embs.T
    _, topk_idx = sims.topk(k, dim=-1)
    correct = (topk_idx == torch.arange(len(query_embs)).unsqueeze(-1)).any(dim=-1)
    return correct.float().mean().item()


def evaluate(model_path: str = "afmrl_trained.pt", device: str = "cpu"):
    model = AFMRL(
        image_dim=128, text_dim=128,
        fusion_dim=256, attr_dim=128,
        embed_dim=128, num_attrs=32,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    Q_img, Q_txt, P_img, P_txt, _, _ = generate_ecommerce_toy_data(
        n_products=1000, image_dim=128, text_dim=128, num_attrs=32, seed=77
    )
    Q_img, Q_txt = Q_img.to(device), Q_txt.to(device)
    P_img, P_txt = P_img.to(device), P_txt.to(device)

    with torch.no_grad():
        q_emb, q_attr_emb, q_attr_logits = model.encode(Q_img, Q_txt)
        p_emb, p_attr_emb, p_attr_logits = model.encode(P_img, P_txt)

    q_emb, p_emb = q_emb.cpu(), p_emb.cpu()

    print("=== AFMRL Evaluation Results ===\n")
    print(f"Recall@1:  {recall_at_k(q_emb, p_emb, k=1):.4f}")
    print(f"Recall@5:  {recall_at_k(q_emb, p_emb, k=5):.4f}")
    print(f"Recall@10: {recall_at_k(q_emb, p_emb, k=10):.4f}")
    print(f"MRR:       {mrr(q_emb, p_emb):.4f}")

    print("\n--- Paper Results (Taobao large-scale e-commerce datasets) ---")
    print("AFMRL achieves SOTA on multiple downstream retrieval tasks.")
    print("Key improvement over VLM2Vec baseline: fine-grained attribute-aware")
    print("embeddings that distinguish highly similar product variants.")
    print("\nAGCL contribution: hard negative mining + false negative filtering")
    print("RAR  contribution: retrieval-reward feedback to attribute generator")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluate(device=device)
