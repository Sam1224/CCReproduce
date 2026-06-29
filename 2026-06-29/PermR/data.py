import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class QueryBatch:
    features: torch.Tensor  # [n_items, d]
    rel_label: torch.Tensor  # [n_items] 0/1
    revenue_label: torch.Tensor  # [n_items] float
    risk_label: torch.Tensor  # [n_items] float (higher = riskier)


class SyntheticRerankDataset(Dataset):
    """A tiny toy dataset for reranking reproduction.

    Each sample corresponds to a "query" with N candidate items.

    We intentionally make revenue and relevance partially misaligned so that
    revenue-max reranking will trade off with relevance unless constrained.
    """

    def __init__(
        self,
        num_queries: int = 200,
        num_items: int = 8,
        feature_dim: int = 16,
        seed: int = 42,
    ) -> None:
        if num_items > 9:
            raise ValueError("For exact ILP-by-enumeration in this repro, set num_items<=9")

        self.num_queries = num_queries
        self.num_items = num_items
        self.feature_dim = feature_dim

        rng = np.random.default_rng(seed)

        # True weights (unknown to the model).
        w_rel = rng.normal(size=(feature_dim,)).astype(np.float32)
        w_rev = rng.normal(size=(feature_dim,)).astype(np.float32)
        w_risk = rng.normal(size=(feature_dim,)).astype(np.float32)

        self._items: List[QueryBatch] = []
        for _ in range(num_queries):
            x = rng.normal(size=(num_items, feature_dim)).astype(np.float32)

            # Relevance probability.
            rel_logit = x @ w_rel + 0.15 * rng.normal(size=(num_items,)).astype(np.float32)
            rel_prob = 1 / (1 + np.exp(-rel_logit))
            rel = (rel_prob > 0.5).astype(np.float32)

            # Revenue: partially aligned with relevance (strong baseline), but still leaves
            # room for constrained reranking to improve monetization.
            rev_raw = (
                0.7 * (x @ w_rel)
                + 0.3 * (x @ w_rev)
                + 0.20 * rng.normal(size=(num_items,)).astype(np.float32)
            )
            rev = np.log1p(np.exp(rev_raw)).astype(np.float32)

            # Risk: also positive.
            risk_raw = x @ w_risk + 0.20 * rng.normal(size=(num_items,)).astype(np.float32)
            risk = (1 / (1 + np.exp(-risk_raw))).astype(np.float32)

            self._items.append(
                QueryBatch(
                    features=torch.from_numpy(x),
                    rel_label=torch.from_numpy(rel),
                    revenue_label=torch.from_numpy(rev),
                    risk_label=torch.from_numpy(risk),
                )
            )

    def __len__(self) -> int:
        return self.num_queries

    def __getitem__(self, idx: int) -> QueryBatch:
        return self._items[idx]


def split_dataset(
    dataset: SyntheticRerankDataset,
    train_ratio: float = 0.8,
    seed: int = 0,
) -> Tuple[List[int], List[int]]:
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    split = int(math.floor(len(indices) * train_ratio))
    return indices[:split], indices[split:]


def collate_queries(batch: List[QueryBatch]) -> Dict[str, torch.Tensor]:
    """Collate multiple query samples.

    We keep batch as [B, N, D] for features, etc.
    """

    features = torch.stack([b.features for b in batch], dim=0)
    rel = torch.stack([b.rel_label for b in batch], dim=0)
    revenue = torch.stack([b.revenue_label for b in batch], dim=0)
    risk = torch.stack([b.risk_label for b in batch], dim=0)
    return {"features": features, "rel": rel, "revenue": revenue, "risk": risk}
