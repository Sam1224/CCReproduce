"""
Toy e-commerce product dataset for TGQ-Former.

Each sample = one product with noisy image features + metadata token ids.
Positive pairs = same product category (same-style I2I retrieval).
"""
import torch
from torch.utils.data import Dataset, DataLoader


class ToyECommerceDataset(Dataset):
    """
    Simulates an e-commerce product dataset for I2I retrieval.

    In production (JD.COM scale):
      - Visual features: ViT / ResNet features from product images
        (often contaminated with promotional overlays)
      - Metadata: tokenized product title + category + attributes
      - Labels: same-style product groups (positive pairs)
    """

    def __init__(
        self,
        num_products: int = 2000,
        visual_dim: int = 2048,
        num_visual_tokens: int = 49,   # 7x7 grid from ViT/ResNet
        meta_seq_len: int = 64,
        vocab_size: int = 30522,
        num_categories: int = 20,
        noise_level: float = 0.3,      # simulates promotional overlay noise
        seed: int = 42,
    ):
        super().__init__()
        torch.manual_seed(seed)
        # Category-based visual features (simulate similar items)
        cat_ids = torch.randint(0, num_categories, (num_products,))
        base_feats = torch.randn(num_categories, visual_dim)
        # Add category-structured signal + noise
        visual = base_feats[cat_ids]
        visual = visual.unsqueeze(1).expand(-1, num_visual_tokens, -1)  # (N, T, D)
        noise = torch.randn(num_products, num_visual_tokens, visual_dim) * noise_level
        self.visual_tokens = visual + noise  # (N, T, D)
        # Metadata tokens (simulated tokenization)
        self.meta_ids = torch.randint(1, vocab_size, (num_products, meta_seq_len))
        # Mask: last 10 tokens are padding
        self.meta_mask = torch.ones(num_products, meta_seq_len, dtype=torch.long)
        self.meta_mask[:, -10:] = 0
        self.category_ids = cat_ids

    def __len__(self):
        return len(self.category_ids)

    def __getitem__(self, idx):
        return {
            "visual_tokens": self.visual_tokens[idx],  # (T, D)
            "meta_ids": self.meta_ids[idx],            # (L,)
            "meta_mask": self.meta_mask[idx],          # (L,)
            "category": self.category_ids[idx],        # scalar
        }


class PairDataset(Dataset):
    """Constructs same-category pairs for contrastive training."""

    def __init__(self, base_dataset: ToyECommerceDataset):
        super().__init__()
        self.ds = base_dataset
        cats = base_dataset.category_ids
        self.pairs = []
        cat_to_indices = {}
        for i, c in enumerate(cats.tolist()):
            cat_to_indices.setdefault(c, []).append(i)
        # Build pairs: for each item, find one positive from same category
        import random
        random.seed(42)
        for i, c in enumerate(cats.tolist()):
            same_cat = [j for j in cat_to_indices[c] if j != i]
            if same_cat:
                j = random.choice(same_cat)
                self.pairs.append((i, j))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        a = self.ds[i]
        b = self.ds[j]
        return {
            "visual_a": a["visual_tokens"],
            "meta_ids_a": a["meta_ids"],
            "meta_mask_a": a["meta_mask"],
            "visual_b": b["visual_tokens"],
            "meta_ids_b": b["meta_ids"],
            "meta_mask_b": b["meta_mask"],
        }


def get_dataloaders(batch_size: int = 32):
    train_base = ToyECommerceDataset(1600, seed=42)
    val_base = ToyECommerceDataset(200, seed=99)
    test_base = ToyECommerceDataset(200, seed=77)

    train_pairs = PairDataset(train_base)
    val_pairs = PairDataset(val_base)

    train_loader = DataLoader(train_pairs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_pairs, batch_size=batch_size)
    test_loader = DataLoader(test_base, batch_size=batch_size)  # for retrieval eval

    return train_loader, val_loader, test_loader, test_base
