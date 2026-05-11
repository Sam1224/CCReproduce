"""
EKTM Training Script
arXiv: 2605.05730

Training pipeline for multi-task CVR prediction with cross-task knowledge transfer.
Supports multiple tasks (e.g., two platform scenarios as described in the paper).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import roc_auc_score
from model import EKTM


def generate_toy_data(
    n_samples: int = 10000,
    input_dim: int = 64,
    num_tasks: int = 2,
    seed: int = 42,
):
    """
    Toy dataset generator with interface aligned to real e-commerce data.

    In production, x would be: concat(user_embedding, item_embedding, context_features)
    labels would be per-task binary CVR labels (did user convert on task k?)
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, input_dim)).astype(np.float32)

    # Simulate correlated but distinct task labels
    # Task 0: main platform CVR (broad)
    # Task 1: secondary platform CVR (more selective)
    latent = X @ rng.standard_normal((input_dim, 1)).astype(np.float32)
    Y = np.zeros((n_samples, num_tasks), dtype=np.float32)
    for k in range(num_tasks):
        noise = rng.standard_normal((n_samples, 1)).astype(np.float32) * 0.5
        logit = latent + noise + rng.standard_normal() * 0.3
        Y[:, k] = (logit.squeeze() > 0).astype(np.float32)

    return X, Y


def compute_gauc(logits: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> float:
    """
    Group AUC (GAUC) — standard metric for recommendation ranking.
    Computes per-user AUC then averages weighted by number of impressions.
    """
    unique_groups = np.unique(groups)
    gauc = 0.0
    total_weight = 0
    for g in unique_groups:
        mask = groups == g
        if labels[mask].sum() == 0 or labels[mask].sum() == mask.sum():
            continue
        auc = roc_auc_score(labels[mask], logits[mask])
        w = mask.sum()
        gauc += auc * w
        total_weight += w
    return gauc / total_weight if total_weight > 0 else 0.0


def train_ektm(
    input_dim: int = 64,
    shared_dim: int = 128,
    hidden_dim: int = 256,
    task_dim: int = 128,
    num_tasks: int = 2,
    n_epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    device: str = "cpu",
):
    print("=== EKTM Training ===")
    print(f"  Tasks: {num_tasks}, Epochs: {n_epochs}, Device: {device}")

    # Data
    X, Y = generate_toy_data(n_samples=20000, input_dim=input_dim, num_tasks=num_tasks)
    n_train = int(0.8 * len(X))
    X_train, X_val = X[:n_train], X[n_train:]
    Y_train, Y_val = Y[:n_train], Y[n_train:]

    train_ds = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(Y_train)
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val), torch.from_numpy(Y_val)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Model
    model = EKTM(
        input_dim=input_dim,
        shared_dim=shared_dim,
        hidden_dim=hidden_dim,
        task_dim=task_dim,
        num_tasks=num_tasks,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    bce_loss = nn.BCEWithLogitsLoss()

    best_val_auc = 0.0

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            out = model(x_batch)
            logits = out["logits"]  # (B, num_tasks)

            # Sum of per-task BCE losses (equal weight, paper uses task-specific weights)
            loss = sum(
                bce_loss(logits[:, k], y_batch[:, k])
                for k in range(num_tasks)
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        all_logits, all_labels = [[] for _ in range(num_tasks)], [[] for _ in range(num_tasks)]
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                out = model(x_batch.to(device))
                lgts = out["logits"].cpu().numpy()
                lbls = y_batch.numpy()
                for k in range(num_tasks):
                    all_logits[k].append(lgts[:, k])
                    all_labels[k].append(lbls[:, k])

        aucs = []
        for k in range(num_tasks):
            lgts_k = np.concatenate(all_logits[k])
            lbls_k = np.concatenate(all_labels[k])
            try:
                auc = roc_auc_score(lbls_k, lgts_k)
            except Exception:
                auc = 0.5
            aucs.append(auc)

        mean_auc = np.mean(aucs)
        if mean_auc > best_val_auc:
            best_val_auc = mean_auc
            torch.save(model.state_dict(), "ektm_best.pt")

        avg_loss = train_loss / len(train_loader)
        auc_str = " | ".join([f"Task{k}: {aucs[k]:.4f}" for k in range(num_tasks)])
        print(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | Val AUC [{auc_str}] | Best: {best_val_auc:.4f}")

    print(f"\nTraining complete. Best validation AUC: {best_val_auc:.4f}")
    return model


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ektm(device=device)
