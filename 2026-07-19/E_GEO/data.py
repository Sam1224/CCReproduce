import hashlib
import numpy as np
import torch
from torch.utils.data import Dataset

HEURISTICS = ["specific", "evidence", "faq", "comparison", "minimal"]
MANIPULATIVE_TERMS = {"guaranteed", "best", "always", "official"}


def stable_hash(text, buckets=128):
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16) % buckets


def bow(text, buckets=128):
    vec = np.zeros(buckets, dtype="float32")
    for token in text.lower().replace(",", " ").replace(".", " ").split():
        vec[stable_hash(token, buckets)] += 1.0
    return torch.tensor(vec / (np.linalg.norm(vec) + 1e-6))


def rewrite_description(description, heuristic):
    if heuristic == "specific":
        return description + " includes measurable specs, use cases, and compatibility details"
    if heuristic == "evidence":
        return description + " explains material, constraints, and verified product facts"
    if heuristic == "faq":
        return description + " answers buyer questions about size, durability, and maintenance"
    if heuristic == "comparison":
        return description + " clarifies tradeoffs against similar alternatives without exaggeration"
    return description


class ToyEGEODataset(Dataset):
    def __init__(self, queries=256, listings_per_query=10, seed=21):
        rng = np.random.default_rng(seed)
        intents = ["durable travel bottle", "comfortable running shoe", "compact office chair", "quiet mechanical keyboard"]
        self.rows = []
        for _ in range(queries):
            intent = str(rng.choice(intents))
            good_idx = int(rng.integers(0, listings_per_query))
            listings = []
            labels = []
            for j in range(listings_per_query):
                quality = float(rng.normal(0.0, 0.4)) + (1.0 if j == good_idx else 0.0)
                desc = f"{intent} product {j} quality {quality:.2f} material details buyer context"
                listings.append(desc)
                labels.append(quality)
            self.rows.append({"query": intent, "listings": listings, "labels": torch.tensor(labels, dtype=torch.float32)})

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        q = bow(row["query"])
        listing_features = torch.stack([bow(x) for x in row["listings"]])
        return {"query_features": q, "listing_features": listing_features, "labels": row["labels"], "listings": row["listings"]}
