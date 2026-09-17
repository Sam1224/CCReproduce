from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import hashlib
import random

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DataConfig:
    vocab_size: int = 512
    max_len: int = 32
    categories: Tuple[str, ...] = ("shoes", "tshirt", "earbuds", "phone_case", "charger", "skincare")
    colors: Tuple[str, ...] = ("red", "blue", "black", "white", "green")
    brands: Tuple[str, ...] = ("acme", "zen", "orbit", "nova")
    synonyms: Dict[str, Tuple[str, ...]] | None = None


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
    digest = hashlib.md5(tok.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) % (vocab_size - 1)) + 1


def encode_tokens(tokens: Iterable[str], cfg: DataConfig) -> torch.Tensor:
    ids = [_hash_token(t, cfg.vocab_size) for t in tokens]
    ids = ids[: cfg.max_len]
    ids += [0] * (cfg.max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


def sample_product(rng: random.Random, cfg: DataConfig) -> Dict[str, str]:
    cat = rng.choice(cfg.categories)
    brand = rng.choice(cfg.brands)
    color = rng.choice(cfg.colors)
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


def sample_query(rng: random.Random, cfg: DataConfig) -> Dict[str, str]:
    cat = rng.choice(cfg.categories)
    color = rng.choice(cfg.colors)
    tokens = [color, cat]
    if cat == "charger" and rng.random() < 0.4:
        tokens = ["iphone", "lightning", "charger"]
    if cat == "earbuds" and rng.random() < 0.5:
        tokens.insert(0, "wireless")
    return {"category": cat, "query": " ".join(tokens)}


def relevance_label(query: Dict[str, str], product: Dict[str, str]) -> int:
    if query["category"] != product["category"]:
        return 0
    q = query["query"].split()
    title = product["title"].split()
    strong = product["color"] in q
    if "wireless" in q:
        strong = strong and product["attr"] == "wireless"
    if "iphone" in q or "lightning" in q:
        strong = strong and (product["category"] == "charger")
    if strong:
        return 3
    overlap = len(set(q) & set(title))
    if overlap >= 2:
        return 2
    return 1


class ToyRelevanceDataset(Dataset):
    def __init__(self, n: int, cfg: DataConfig, seed: int = 0):
        self.cfg = cfg
        self.items: List[Dict[str, torch.Tensor]] = []
        rng = random.Random(seed)
        for _ in range(n):
            q = sample_query(rng, cfg)
            p = sample_product(rng, cfg)
            y = relevance_label(q, p)
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


@dataclass(frozen=True)
class EvalMetrics:
    acc: float
    macro_f1: float
    win_rate: float
    relevant_precision: float
    relevant_recall: float


def _macro_f1(tp_fp_fn_by_class):
    f1s = []
    for tp, fp, fn in tp_fp_fn_by_class:
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        f1 = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
        f1s.append(f1)
    return float(sum(f1s) / len(f1s))


@torch.no_grad()
def evaluate(model, ds, batch_size: int = 128) -> EvalMetrics:
    model.eval()
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
    correct = 0
    total = 0
    tp_fp_fn = [[0, 0, 0] for _ in range(4)]
    win = 0
    rel_tp = rel_fp = rel_fn = 0
    for batch in loader:
        out = model(batch["q_ids"], batch["d_ids"])
        pred = out["coarse_logits"].argmax(dim=-1)
        gold = batch["y"]
        total += int(gold.numel())
        correct += int((pred == gold).sum().item())
        for c in range(4):
            pc = pred == c
            gc = gold == c
            tp_fp_fn[c][0] += int((pc & gc).sum().item())
            tp_fp_fn[c][1] += int((pc & ~gc).sum().item())
            tp_fp_fn[c][2] += int((~pc & gc).sum().item())
        is_good = gold >= 2
        pred_good = pred >= 2
        rel_tp += int((pred_good & is_good).sum().item())
        rel_fp += int((pred_good & ~is_good).sum().item())
        rel_fn += int((~pred_good & is_good).sum().item())
        win += int((pred_good == is_good).sum().item())
    precision = rel_tp / max(1, rel_tp + rel_fp)
    recall = rel_tp / max(1, rel_tp + rel_fn)
    return EvalMetrics(
        acc=correct / max(1, total),
        macro_f1=_macro_f1(tp_fp_fn),
        win_rate=win / max(1, total),
        relevant_precision=precision,
        relevant_recall=recall,
    )


def build_cfg_from_ckpt(payload: Dict) -> DataConfig:
    return DataConfig(**payload["data_cfg"])
