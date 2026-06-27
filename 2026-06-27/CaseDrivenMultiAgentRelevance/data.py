from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import random

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ToyRelevanceConfig:
    vocab_size: int = 512
    max_len: int = 32

    # fixed small catalog
    categories: Tuple[str, ...] = (
        "shoes",
        "tshirt",
        "earbuds",
        "phone_case",
        "charger",
        "skincare",
    )

    colors: Tuple[str, ...] = ("red", "blue", "black", "white", "green")
    brands: Tuple[str, ...] = ("acme", "zen", "orbit", "nova")

    # knowledge / synonym hooks (used by DeepSearchAgent-style expansion)
    synonyms: Dict[str, Tuple[str, ...]] = None  # filled in __post_init__


def default_synonyms() -> Dict[str, Tuple[str, ...]]:
    return {
        "iphone": ("ios", "apple"),
        "charger": ("power_adapter", "charging_brick"),
        "lightning": ("iphone", "ios"),
        "wireless": ("bluetooth",),
        "earbuds": ("headphones", "earphones"),
        "tshirt": ("tee", "t-shirt"),
    }


def _hash_token(tok: str, vocab_size: int) -> int:
    # stable-ish mapping for toy use
    return (abs(hash(tok)) % (vocab_size - 1)) + 1


def encode_tokens(tokens: Iterable[str], cfg: ToyRelevanceConfig) -> torch.Tensor:
    ids = [_hash_token(t, cfg.vocab_size) for t in tokens]
    ids = ids[: cfg.max_len]
    ids += [0] * (cfg.max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


def sample_product(rng: random.Random, cfg: ToyRelevanceConfig) -> Dict[str, str]:
    cat = rng.choice(cfg.categories)
    brand = rng.choice(cfg.brands)
    color = rng.choice(cfg.colors)
    # a couple simple attributes
    attr = "wireless" if cat in ("earbuds", "charger") and rng.random() < 0.5 else ""
    size = "" if cat not in ("shoes", "tshirt") else rng.choice(("s", "m", "l"))

    title_tokens = [brand, color, cat]
    if attr:
        title_tokens.insert(1, attr)
    if size:
        title_tokens.append(size)

    return {
        "category": cat,
        "brand": brand,
        "color": color,
        "attr": attr,
        "size": size,
        "title": " ".join([t for t in title_tokens if t]),
    }


def sample_query(rng: random.Random, cfg: ToyRelevanceConfig) -> Dict[str, str]:
    # introduce "implicit" / knowledge-driven tokens sometimes
    cat = rng.choice(cfg.categories)
    color = rng.choice(cfg.colors)

    tokens = [color, cat]

    if cat == "charger" and rng.random() < 0.4:
        tokens = ["iphone", "lightning", "charger"]

    if cat == "earbuds" and rng.random() < 0.5:
        tokens.insert(0, "wireless")

    return {"category": cat, "query": " ".join(tokens)}


def relevance_label(query: Dict[str, str], product: Dict[str, str]) -> int:
    # 0..3 as in the paper: Irrelevant / Weak / Relevant / Strong.
    if query["category"] != product["category"]:
        return 0

    q = query["query"].split()
    title = product["title"].split()

    # strong: category match + color match + (wireless/iphone implied) matches
    strong = product["color"] in q
    if "wireless" in q:
        strong = strong and product["attr"] == "wireless"

    if "iphone" in q or "lightning" in q:
        # toy: treat "iphone/lightning" as a proxy for brand affinity
        strong = strong and (product["category"] == "charger")

    if strong:
        return 3

    # relevant: category match + at least one attribute token overlaps
    overlap = len(set(q) & set(title))
    if overlap >= 2:
        return 2

    # weakly relevant otherwise
    return 1


class ToyRelevanceDataset(Dataset):
    def __init__(self, n: int, cfg: ToyRelevanceConfig, seed: int = 0):
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.items: List[Dict[str, torch.Tensor]] = []
        syn = cfg.synonyms or default_synonyms()

        for _ in range(n):
            q = sample_query(self.rng, cfg)
            p = sample_product(self.rng, cfg)
            y = relevance_label(q, p)

            # expose both raw text and encoded ids
            q_tokens = q["query"].split()
            p_tokens = p["title"].split()

            self.items.append(
                {
                    "q_text": q["query"],
                    "d_text": p["title"],
                    "q_ids": encode_tokens(q_tokens, cfg),
                    "d_ids": encode_tokens(p_tokens, cfg),
                    "y": torch.tensor(y, dtype=torch.long),
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.items[idx]
