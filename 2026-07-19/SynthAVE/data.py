import hashlib
import numpy as np
import torch
from torch.utils.data import Dataset

LANGUAGES = ["es", "fr", "it", "de"]
PRODUCT_TYPES = ["shoe", "phone", "sofa", "watch", "bottle", "jacket"]
ATTRIBUTES = ["color", "material", "size", "capacity"]
VALUES = ["red", "blue", "black", "cotton", "leather", "small", "large", "500ml"]


def stable_hash(text, buckets=256):
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % buckets


def featurize(text, buckets=256):
    vec = np.zeros(buckets, dtype="float32")
    for token in text.lower().replace("/", " ").split():
        vec[stable_hash(token, buckets)] += 1.0
    norm = np.linalg.norm(vec) + 1e-6
    return torch.tensor(vec / norm)


class ToyAVEDataset(Dataset):
    def __init__(self, size=1024, seed=11, noise=0.12):
        rng = np.random.default_rng(seed)
        self.rows = []
        for _ in range(size):
            lang = rng.choice(LANGUAGES)
            product = rng.choice(PRODUCT_TYPES)
            attr = rng.choice(ATTRIBUTES)
            value_id = int(rng.integers(0, len(VALUES)))
            value = VALUES[value_id]
            title = f"{lang} {product} premium {attr} {value} catalog description"
            synthetic_label = value_id
            if rng.random() < noise:
                synthetic_label = int(rng.integers(0, len(VALUES)))
            self.rows.append({
                "features": featurize(title),
                "value_id": torch.tensor(value_id, dtype=torch.long),
                "synthetic_label": torch.tensor(synthetic_label, dtype=torch.long),
                "language": lang,
                "attribute": attr,
                "text": title,
            })

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]
