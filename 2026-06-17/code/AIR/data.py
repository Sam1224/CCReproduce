"""
AIR (Atomic Intent Reasoning) — Data utilities
Toy dataset simulating a content-to-e-commerce cross-domain scenario
(content domain: video watch history; commerce domain: purchase history)
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional


# ── Toy data generation ────────────────────────────────────────────────────

def generate_toy_data(
    n_users: int = 1000,
    n_content_items: int = 5000,
    n_commerce_items: int = 3000,
    n_atoms: int = 128,          # number of atomic intent categories
    seq_len: int = 20,
    seed: int = 42,
) -> Dict:
    """Generate synthetic cross-domain interaction data."""
    rng = np.random.default_rng(seed)

    # User latent factors (hidden "true" intent vectors)
    user_intents = rng.standard_normal((n_users, 64)).astype(np.float32)

    # Content item embeddings (text + visual features, dim=128)
    content_embs = rng.standard_normal((n_content_items, 128)).astype(np.float32)

    # Commerce item embeddings
    commerce_embs = rng.standard_normal((n_commerce_items, 128)).astype(np.float32)

    # Atomic intent prototypes (learned offline by LLM — here we simulate)
    atom_prototypes = rng.standard_normal((n_atoms, 128)).astype(np.float32)
    atom_prototypes /= np.linalg.norm(atom_prototypes, axis=1, keepdims=True)

    # Generate user content sequences (simulate watch history)
    content_seqs = []
    for u in range(n_users):
        # User prefers items whose embeddings are close to their intent
        scores = content_embs @ user_intents[u][:64]
        probs = np.exp(scores / 0.5)
        probs /= probs.sum()
        seq = rng.choice(n_content_items, size=seq_len, p=probs, replace=True)
        content_seqs.append(seq)
    content_seqs = np.array(content_seqs)

    # Generate user purchase sequences (commerce domain)
    commerce_seqs = []
    commerce_labels = []
    for u in range(n_users):
        scores = commerce_embs @ user_intents[u][:64]
        probs = np.exp(scores / 0.5)
        probs /= probs.sum()
        seq = rng.choice(n_commerce_items, size=seq_len - 1, p=probs, replace=True)
        label = rng.choice(n_commerce_items, p=probs)
        commerce_seqs.append(seq)
        commerce_labels.append(label)
    commerce_seqs = np.array(commerce_seqs)
    commerce_labels = np.array(commerce_labels)

    return {
        "n_users": n_users,
        "n_content_items": n_content_items,
        "n_commerce_items": n_commerce_items,
        "n_atoms": n_atoms,
        "user_intents": user_intents,
        "content_embs": content_embs,
        "commerce_embs": commerce_embs,
        "atom_prototypes": atom_prototypes,
        "content_seqs": content_seqs,
        "commerce_seqs": commerce_seqs,
        "commerce_labels": commerce_labels,
    }


# ── Offline atomic intent assignment ───────────────────────────────────────

def assign_atomic_intents(
    content_seqs: np.ndarray,
    content_embs: np.ndarray,
    atom_prototypes: np.ndarray,
    top_k: int = 5,
) -> np.ndarray:
    """
    Simulate offline LLM-based atomic intent mining.
    Each user's content history is represented as a weighted sum of atom prototypes
    (in the real AIR paper, an LLM reasons over natural-language behavior descriptions
    to produce soft intent assignments).

    Returns:
        user_atom_weights: (n_users, n_atoms) sparse weight matrix
    """
    n_users = content_seqs.shape[0]
    n_atoms = atom_prototypes.shape[0]
    user_atom_weights = np.zeros((n_users, n_atoms), dtype=np.float32)

    for u in range(n_users):
        # Average content embedding for user
        user_avg = content_embs[content_seqs[u]].mean(axis=0)
        user_avg /= (np.linalg.norm(user_avg) + 1e-8)

        # Similarity to each atom prototype
        sims = atom_prototypes @ user_avg
        # Sparse: keep top-k atoms
        topk_idx = np.argsort(sims)[-top_k:]
        weights = np.exp(sims[topk_idx])
        weights /= weights.sum()
        user_atom_weights[u, topk_idx] = weights

    return user_atom_weights


# ── PyTorch Dataset ─────────────────────────────────────────────────────────

class CrossDomainDataset(Dataset):
    """
    Each sample: (user_id, content_seq, commerce_seq, commerce_label)
    """
    def __init__(
        self,
        user_ids: np.ndarray,
        content_seqs: np.ndarray,
        commerce_seqs: np.ndarray,
        commerce_labels: np.ndarray,
    ):
        self.user_ids = torch.LongTensor(user_ids)
        self.content_seqs = torch.LongTensor(content_seqs)
        self.commerce_seqs = torch.LongTensor(commerce_seqs)
        self.commerce_labels = torch.LongTensor(commerce_labels)

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        return {
            "user_id": self.user_ids[idx],
            "content_seq": self.content_seqs[idx],
            "commerce_seq": self.commerce_seqs[idx],
            "label": self.commerce_labels[idx],
        }


def build_dataloaders(data: Dict, train_ratio=0.8, batch_size=256):
    n = data["n_users"]
    all_ids = np.arange(n)
    split = int(n * train_ratio)
    train_ids, val_ids = all_ids[:split], all_ids[split:]

    def make_loader(ids, shuffle):
        ds = CrossDomainDataset(
            ids,
            data["content_seqs"][ids],
            data["commerce_seqs"][ids],
            data["commerce_labels"][ids],
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    return make_loader(train_ids, True), make_loader(val_ids, False)


if __name__ == "__main__":
    print("Generating toy dataset...")
    data = generate_toy_data()
    print(f"  Users: {data['n_users']}")
    print(f"  Content items: {data['n_content_items']}")
    print(f"  Commerce items: {data['n_commerce_items']}")
    print(f"  Atomic intents: {data['n_atoms']}")

    print("Assigning atomic intents...")
    atom_weights = assign_atomic_intents(
        data["content_seqs"], data["content_embs"], data["atom_prototypes"]
    )
    print(f"  Atom weight matrix shape: {atom_weights.shape}")
    print(f"  Mean non-zero atoms per user: {(atom_weights > 0).sum(1).mean():.1f}")

    train_loader, val_loader = build_dataloaders(data)
    batch = next(iter(train_loader))
    print(f"  Batch keys: {list(batch.keys())}")
    print(f"  content_seq shape: {batch['content_seq'].shape}")
    print("Data module OK.")
