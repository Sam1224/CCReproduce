"""
AFMRL Training Script
arXiv: 2604.20135

Two-stage training:
  Stage 1 (AGCL): Train encoder + attribute generator with attribute-guided contrastive loss
  Stage 2 (RAR):  Fine-tune attribute generator with retrieval-reward signal
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from model import AFMRL, AGCLLoss, RARLoss


def generate_ecommerce_toy_data(
    n_products: int = 5000,
    image_dim: int = 768,
    text_dim: int = 768,
    num_attrs: int = 64,
    num_neg: int = 8,
    seed: int = 42,
):
    """
    Toy e-commerce product dataset.

    Each sample = (query_product, positive_product, N negative_products)
    where positives are "same product" pairs (identical product retrieval task).

    In production:
      image_feat = ViT / ResNet features from product images
      text_feat  = CLIP / BERT features from product titles/descriptions
    """
    rng = np.random.default_rng(seed)

    # Simulate product features with attribute-based clustering
    num_clusters = 50  # 50 product categories
    cluster_centers_img = rng.standard_normal((num_clusters, image_dim)).astype(np.float32)
    cluster_centers_txt = rng.standard_normal((num_clusters, text_dim)).astype(np.float32)

    labels = rng.integers(0, num_clusters, size=n_products)
    image_feats = (
        cluster_centers_img[labels]
        + rng.standard_normal((n_products, image_dim)).astype(np.float32) * 0.3
    )
    text_feats = (
        cluster_centers_txt[labels]
        + rng.standard_normal((n_products, text_dim)).astype(np.float32) * 0.3
    )

    # Build pairs: (query, positive, N negatives)
    Q_img = image_feats[:n_products // 2]
    Q_txt = text_feats[:n_products // 2]
    P_img = image_feats[:n_products // 2] + rng.standard_normal(
        (n_products // 2, image_dim)
    ).astype(np.float32) * 0.1  # small perturbation = same product
    P_txt = text_feats[:n_products // 2] + rng.standard_normal(
        (n_products // 2, text_dim)
    ).astype(np.float32) * 0.1

    # Random negatives from other products
    N_img = np.stack([image_feats[n_products // 2:][rng.choice(n_products // 2, size=num_neg)] for _ in range(n_products // 2)]).astype(np.float32)
    N_txt = np.stack([text_feats[n_products // 2:][rng.choice(n_products // 2, size=num_neg)] for _ in range(n_products // 2)]).astype(np.float32)

    return (
        torch.from_numpy(Q_img), torch.from_numpy(Q_txt),
        torch.from_numpy(P_img), torch.from_numpy(P_txt),
        torch.from_numpy(N_img), torch.from_numpy(N_txt),
    )


def recall_at_k(query_embs: torch.Tensor, pos_embs: torch.Tensor, k: int = 10) -> float:
    """Compute Recall@K for retrieval evaluation."""
    sims = query_embs @ pos_embs.T  # (B, B)
    _, topk_idx = sims.topk(k, dim=-1)
    correct = (topk_idx == torch.arange(len(query_embs)).unsqueeze(-1)).any(dim=-1)
    return correct.float().mean().item()


def train_stage1_agcl(
    model: AFMRL,
    train_loader: DataLoader,
    n_epochs: int = 15,
    lr: float = 1e-4,
    device: str = "cpu",
) -> AFMRL:
    """Stage 1: Attribute-Guided Contrastive Learning."""
    print("=== Stage 1: AGCL Training ===")
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    agcl_loss_fn = AGCLLoss(temperature=0.07, tau_fn=0.7, top_hard_k=8)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for q_img, q_txt, p_img, p_txt, n_img, n_txt in train_loader:
            q_img, q_txt = q_img.to(device), q_txt.to(device)
            p_img, p_txt = p_img.to(device), p_txt.to(device)
            n_img, n_txt = n_img.to(device), n_txt.to(device)

            optimizer.zero_grad()
            out = model(q_img, q_txt, p_img, p_txt, n_img, n_txt)

            loss = agcl_loss_fn(
                query_emb=out["q_emb"],
                pos_emb=out["p_emb"],
                neg_embs=out["neg_embs"],
                query_attr=out["q_attr_logits"],
                neg_attrs=out["neg_attr_logits"],
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        print(f"  Epoch {epoch+1:02d} | AGCL Loss: {epoch_loss / len(train_loader):.4f}")

    return model


def train_stage2_rar(
    model: AFMRL,
    train_loader: DataLoader,
    n_epochs: int = 10,
    lr: float = 5e-5,
    device: str = "cpu",
) -> AFMRL:
    """Stage 2: Retrieval-aware Attribute Reinforcement."""
    print("\n=== Stage 2: RAR Fine-tuning ===")
    # Only fine-tune the attribute generator — freeze encoder + repr head
    for name, param in model.named_parameters():
        if "attr_gen" in name or "attr_to_embed" in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-5,
    )
    rar_loss_fn = RARLoss(reward_scale=0.1)

    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for q_img, q_txt, p_img, p_txt, _, _ in train_loader:
            q_img, q_txt = q_img.to(device), q_txt.to(device)
            p_img, p_txt = p_img.to(device), p_txt.to(device)

            optimizer.zero_grad()
            out = model(q_img, q_txt, p_img, p_txt)

            loss = rar_loss_fn(
                attr_emb=out["q_attr_emb"],
                pos_emb=out["p_emb"],
                attr_proj=out["attr_to_embed"],
            )
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"  Epoch {epoch+1:02d} | RAR Loss: {epoch_loss / len(train_loader):.6f}")

    # Unfreeze all parameters
    for param in model.parameters():
        param.requires_grad_(True)

    return model


def train_afmrl(
    image_dim: int = 128,   # reduced for toy data
    text_dim: int = 128,
    fusion_dim: int = 256,
    attr_dim: int = 128,
    embed_dim: int = 128,
    num_attrs: int = 32,
    num_neg: int = 8,
    batch_size: int = 64,
    device: str = "cpu",
):
    print("=== AFMRL Two-Stage Training ===")
    print(f"  Device: {device}")

    Q_img, Q_txt, P_img, P_txt, N_img, N_txt = generate_ecommerce_toy_data(
        n_products=2000, image_dim=image_dim, text_dim=text_dim,
        num_attrs=num_attrs, num_neg=num_neg,
    )
    ds = TensorDataset(Q_img, Q_txt, P_img, P_txt, N_img, N_txt)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = AFMRL(
        image_dim=image_dim, text_dim=text_dim,
        fusion_dim=fusion_dim, attr_dim=attr_dim,
        embed_dim=embed_dim, num_attrs=num_attrs,
    ).to(device)

    # Stage 1: AGCL
    model = train_stage1_agcl(model, loader, n_epochs=10, device=device)

    # Evaluate after Stage 1
    model.eval()
    with torch.no_grad():
        q_emb_all, p_emb_all = [], []
        for q_img, q_txt, p_img, p_txt, _, _ in loader:
            q_e, _, _ = model.encode(q_img.to(device), q_txt.to(device))
            p_e, _, _ = model.encode(p_img.to(device), p_txt.to(device))
            q_emb_all.append(q_e.cpu())
            p_emb_all.append(p_e.cpu())
        q_all = torch.cat(q_emb_all)
        p_all = torch.cat(p_emb_all)
        r1 = recall_at_k(q_all, p_all, k=1)
        r10 = recall_at_k(q_all, p_all, k=10)
    print(f"\nAfter Stage 1 (AGCL): Recall@1={r1:.4f}, Recall@10={r10:.4f}")

    # Stage 2: RAR
    model = train_stage2_rar(model, loader, n_epochs=5, device=device)

    # Evaluate after Stage 2
    model.eval()
    with torch.no_grad():
        q_emb_all, p_emb_all = [], []
        for q_img, q_txt, p_img, p_txt, _, _ in loader:
            q_e, _, _ = model.encode(q_img.to(device), q_txt.to(device))
            p_e, _, _ = model.encode(p_img.to(device), p_txt.to(device))
            q_emb_all.append(q_e.cpu())
            p_emb_all.append(p_e.cpu())
        q_all = torch.cat(q_emb_all)
        p_all = torch.cat(p_emb_all)
        r1 = recall_at_k(q_all, p_all, k=1)
        r10 = recall_at_k(q_all, p_all, k=10)
    print(f"After Stage 2 (RAR):  Recall@1={r1:.4f}, Recall@10={r10:.4f}")

    torch.save(model.state_dict(), "afmrl_trained.pt")
    print("\nModel saved to afmrl_trained.pt")
    return model


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_afmrl(device=device)
