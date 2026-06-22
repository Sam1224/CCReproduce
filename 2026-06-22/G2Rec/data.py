"""Toy synthetic dataset for G2Rec (Sec. 2.1 notations + Problem 1).

We synthesize item embeddings X with latent cluster structure and user
interaction sequences I_u that mostly draw from each user's preferred latent
clusters, so the item-item co-engagement graph has meaningful community
structure for soft clustering. Leave-one-out split (Sec. 5.1): last item =
test, second-to-last = val, rest = train.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

PAD = 0  # reserved padding item id; real items are 1..num_items


def build_toy_data(num_users=300, num_items=200, d=64, n_latent=8,
                   min_len=6, max_len=15, seed=0):
    rng = np.random.default_rng(seed)
    # latent cluster centers + per-item embeddings (item ids are 1-indexed)
    centers = rng.normal(0, 1.0, size=(n_latent, d))
    item_cluster = rng.integers(0, n_latent, size=num_items)
    X = centers[item_cluster] + rng.normal(0, 0.35, size=(num_items, d))
    X = X.astype(np.float32)
    # group items by latent cluster
    cluster_items = {c: np.where(item_cluster == c)[0] + 1 for c in range(n_latent)}
    sequences = []
    for _ in range(num_users):
        prefs = rng.choice(n_latent, size=rng.integers(1, 4), replace=False)
        length = int(rng.integers(min_len, max_len + 1))
        seq = []
        for _ in range(length):
            if rng.random() < 0.85:                      # in-interest item
                c = rng.choice(prefs)
                seq.append(int(rng.choice(cluster_items[c])))
            else:                                        # exploration item
                seq.append(int(rng.integers(1, num_items + 1)))
        sequences.append(seq)
    # prepend PAD row so X is indexable by 1-based item id
    X_full = np.vstack([np.zeros((1, d), np.float32), X])
    return X_full, sequences, num_items


class SeqDataset(Dataset):
    """Leave-one-out sequential recommendation dataset."""

    def __init__(self, sequences, split="train", num_items=200):
        self.split = split
        self.num_items = num_items
        self.samples = []
        for seq in sequences:
            if len(seq) < 3:
                continue
            if split == "train":
                self.samples.append(seq[:-2])            # all but last two
            elif split == "valid":
                self.samples.append((seq[:-2], seq[-2]))  # history -> val target
            else:                                         # test
                self.samples.append((seq[:-1], seq[-1]))  # history -> test target

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_train(batch):
    """Pad train sequences; returns (seqs, lengths)."""
    lengths = [len(s) for s in batch]
    L = max(lengths)
    out = torch.zeros(len(batch), L, dtype=torch.long)
    for i, s in enumerate(batch):
        out[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    return out, torch.tensor(lengths)


def collate_eval(batch):
    """Pad eval histories; returns (hist, lengths, targets)."""
    hists = [b[0] for b in batch]
    targets = torch.tensor([b[1] for b in batch], dtype=torch.long)
    lengths = [len(h) for h in hists]
    L = max(lengths)
    out = torch.zeros(len(batch), L, dtype=torch.long)
    for i, h in enumerate(hists):
        out[i, :len(h)] = torch.tensor(h, dtype=torch.long)
    return out, torch.tensor(lengths), targets


def make_loaders(sequences, num_items, batch_size=32):
    tr = DataLoader(SeqDataset(sequences, "train", num_items), batch_size=batch_size,
                    shuffle=True, collate_fn=collate_train)
    va = DataLoader(SeqDataset(sequences, "valid", num_items), batch_size=batch_size,
                    shuffle=False, collate_fn=collate_eval)
    te = DataLoader(SeqDataset(sequences, "test", num_items), batch_size=batch_size,
                    shuffle=False, collate_fn=collate_eval)
    return tr, va, te


if __name__ == "__main__":
    X, seqs, n = build_toy_data()
    print("X", X.shape, "num_users", len(seqs), "num_items", n)
    print("example seq", seqs[0])
