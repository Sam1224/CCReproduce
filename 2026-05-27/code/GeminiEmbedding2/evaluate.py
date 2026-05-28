"""
Evaluation script for Gemini Embedding 2 reproduction.

Evaluates on standard retrieval benchmarks (toy versions):
  - Text-to-text retrieval (BEIR-style)
  - Image-text retrieval (COCO-style)
  - Zero-shot classification via embedding similarity

Usage:
    python evaluate.py --checkpoint checkpoints/stage1_final.pt
    python evaluate.py --checkpoint checkpoints/stage2_final.pt --task retrieval
"""

import argparse
import logging
import os
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from model import GE2Config, GeminiEmbeddingModel
from dataset import TextPairDataset, collate_text_pair
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def recall_at_k(sims: torch.Tensor, k: int) -> float:
    """sims: (N, M) similarity matrix. Ground truth: diagonal."""
    N = sims.size(0)
    ranks = sims.argsort(dim=1, descending=True)
    labels = torch.arange(N, device=sims.device)
    hits = (ranks[:, :k] == labels.unsqueeze(1)).any(dim=1).float()
    return hits.mean().item()


def ndcg_at_k(sims: torch.Tensor, k: int) -> float:
    """Compute NDCG@K assuming single relevant document per query."""
    ranks = sims.argsort(dim=1, descending=True)
    labels = torch.arange(sims.size(0), device=sims.device)
    positions = (ranks == labels.unsqueeze(1)).float().argmax(dim=1) + 1
    dcg = 1.0 / torch.log2(positions.float() + 1)
    # Ideal DCG = 1 (single relevant at position 1)
    ndcg = (positions <= k).float() * dcg
    return ndcg.mean().item()


def mean_reciprocal_rank(sims: torch.Tensor) -> float:
    ranks = sims.argsort(dim=1, descending=True)
    labels = torch.arange(sims.size(0), device=sims.device)
    positions = (ranks == labels.unsqueeze(1)).float().argmax(dim=1) + 1
    return (1.0 / positions.float()).mean().item()


# ---------------------------------------------------------------------------
# Evaluation tasks
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_text_retrieval(model: GeminiEmbeddingModel,
                             device: torch.device,
                             num_queries: int = 200) -> Dict[str, float]:
    """Text-to-text retrieval evaluation (toy BEIR-style)."""
    model.eval()
    from model import GE2Config
    cfg = model.cfg

    dataset = TextPairDataset(num_samples=num_queries, vocab_size=cfg.vocab_size,
                              seq_len=128)
    loader = DataLoader(dataset, batch_size=64, collate_fn=collate_text_pair)

    q_embeds, k_embeds = [], []
    for batch in loader:
        q_ids = batch["query_ids"].to(device)
        q_mask = batch["query_mask"].to(device)
        k_ids = batch["key_ids"].to(device)
        k_mask = batch["key_mask"].to(device)
        q_emb = model(model.encode_text(q_ids), q_mask)
        k_emb = model(model.encode_text(k_ids), k_mask)
        q_embeds.append(q_emb)
        k_embeds.append(k_emb)

    Q = torch.cat(q_embeds)
    K = torch.cat(k_embeds)
    sims = torch.matmul(Q, K.T)

    return {
        "R@1": recall_at_k(sims, 1),
        "R@5": recall_at_k(sims, 5),
        "R@10": recall_at_k(sims, 10),
        "NDCG@10": ndcg_at_k(sims, 10),
        "MRR": mean_reciprocal_rank(sims),
    }


@torch.no_grad()
def evaluate_cross_modal_retrieval(model: GeminiEmbeddingModel,
                                   device: torch.device,
                                   num_samples: int = 100) -> Dict[str, float]:
    """
    Cross-modal (image→text) retrieval evaluation.
    Simulates MSCOCO-style evaluation.
    """
    model.eval()
    cfg = model.cfg
    # Toy: generate random image patches and paired text
    B = num_samples
    num_patches = (cfg.image_size // cfg.image_patch_size) ** 2
    patch_dim = 3 * cfg.image_patch_size ** 2

    torch.manual_seed(100)
    patches = torch.randn(B, num_patches, patch_dim).to(device)
    text_ids = torch.randint(1, cfg.vocab_size, (B, 64)).to(device)
    text_mask = torch.ones(B, 64, dtype=torch.long).to(device)
    patch_mask = torch.ones(B, num_patches + 1, dtype=torch.long).to(device)

    # Batch forward
    bs = 32
    img_embeds, txt_embeds = [], []
    for i in range(0, B, bs):
        p = patches[i:i+bs]
        pm = patch_mask[i:i+bs]
        t = text_ids[i:i+bs]
        tm = text_mask[i:i+bs]
        img_embeds.append(model(model.encode_image(p), pm))
        txt_embeds.append(model(model.encode_text(t), tm))

    I = torch.cat(img_embeds)
    T = torch.cat(txt_embeds)
    sims_i2t = torch.matmul(I, T.T)
    sims_t2i = sims_i2t.T

    return {
        "Image→Text R@1": recall_at_k(sims_i2t, 1),
        "Image→Text R@5": recall_at_k(sims_i2t, 5),
        "Text→Image R@1": recall_at_k(sims_t2i, 1),
        "Text→Image R@5": recall_at_k(sims_t2i, 5),
        "COCO-style R@1 avg": (recall_at_k(sims_i2t, 1) + recall_at_k(sims_t2i, 1)) / 2,
    }


@torch.no_grad()
def evaluate_clustering(model: GeminiEmbeddingModel,
                        device: torch.device) -> Dict[str, float]:
    """
    Clustering evaluation: embed texts from K classes, measure intra/inter
    class cosine similarity ratio (proxy for clustering quality).
    """
    model.eval()
    cfg = model.cfg
    K, n_per_class, seq_len = 10, 20, 64
    torch.manual_seed(200)
    # Generate class-specific token patterns (each class has a distinct prefix)
    class_prefix = torch.randint(1, cfg.vocab_size, (K, 8))
    all_embeds, all_labels = [], []
    for cls in range(K):
        prefix = class_prefix[cls].unsqueeze(0).expand(n_per_class, -1)
        suffix = torch.randint(1, cfg.vocab_size, (n_per_class, seq_len - 8))
        ids = torch.cat([prefix, suffix], dim=1).to(device)
        mask = torch.ones(n_per_class, seq_len, dtype=torch.long).to(device)
        emb = model(model.encode_text(ids), mask)
        all_embeds.append(emb)
        all_labels.extend([cls] * n_per_class)
    embeds = torch.cat(all_embeds)
    labels = torch.tensor(all_labels, device=device)
    sims = torch.matmul(embeds, embeds.T)

    # Intra-class vs inter-class similarity
    same_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    diff_mask = 1.0 - same_mask
    eye = torch.eye(len(labels), device=device)
    intra_sim = ((sims * same_mask * (1 - eye)).sum() /
                 (same_mask * (1 - eye)).sum().clamp(min=1)).item()
    inter_sim = ((sims * diff_mask).sum() / diff_mask.sum().clamp(min=1)).item()
    return {
        "intra_class_sim": intra_sim,
        "inter_class_sim": inter_sim,
        "sim_ratio": intra_sim / max(abs(inter_sim), 1e-6),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--task", type=str, default="all",
                        choices=["all", "retrieval", "cross_modal", "clustering"])
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = GE2Config(
        hidden_dim=args.hidden_dim,
        num_heads=max(1, args.hidden_dim // 64),
        num_layers=args.num_layers,
        ffn_dim=args.hidden_dim * 4,
        embed_dim=args.hidden_dim,
    )
    model = GeminiEmbeddingModel(cfg).to(device)

    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        log.info(f"Loaded checkpoint: {args.checkpoint}")
    else:
        log.info("No checkpoint loaded — evaluating random-init model (toy baseline)")

    model.eval()

    if args.task in ("all", "retrieval"):
        log.info("--- Text Retrieval ---")
        metrics = evaluate_text_retrieval(model, device)
        for k, v in metrics.items():
            log.info(f"  {k}: {v:.4f}")

    if args.task in ("all", "cross_modal"):
        log.info("--- Cross-Modal Retrieval (Image↔Text) ---")
        metrics = evaluate_cross_modal_retrieval(model, device)
        for k, v in metrics.items():
            log.info(f"  {k}: {v:.4f}")

    if args.task in ("all", "clustering"):
        log.info("--- Clustering Quality ---")
        metrics = evaluate_clustering(model, device)
        for k, v in metrics.items():
            log.info(f"  {k}: {v:.4f}")

    log.info("Done.")


if __name__ == "__main__":
    main()
