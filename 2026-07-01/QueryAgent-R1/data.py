from __future__ import annotations

import random
import re
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

_WORD_RE = re.compile(r"[a-z0-9]+")

CATEGORIES = ["phone", "dress", "coffee", "camera", "headphones", "toy", "sofa", "laptop"]
COLORS = ["black", "white", "red", "blue", "green", "pink"]
INTENTS = ["budget", "premium", "portable", "gift", "business", "gaming"]


def tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


class Vocab:
    def __init__(self, texts: list[str]):
        toks = ["[pad]", "[unk]"]
        seen = set(toks)
        for text in texts:
            for tok in tokenize(text):
                if tok not in seen:
                    toks.append(tok)
                    seen.add(tok)
        self.itos = toks
        self.stoi = {t: i for i, t in enumerate(toks)}

    def encode(self, text: str, max_len: int = 12) -> torch.Tensor:
        ids = [self.stoi.get(t, 1) for t in tokenize(text)[:max_len]]
        ids += [0] * (max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)


@dataclass(frozen=True)
class Product:
    pid: int
    category: str
    color: str
    intent: str
    price: float

    @property
    def title(self) -> str:
        return f"{self.color} {self.intent} {self.category}"


def make_products(n: int = 160, seed: int = 0) -> list[Product]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        out.append(Product(i, rng.choice(CATEGORIES), rng.choice(COLORS), rng.choice(INTENTS), round(rng.uniform(8, 999), 2)))
    return out


def product_query(p: Product) -> str:
    return f"{p.intent} {p.color} {p.category}"


class QueryAgentDataset(Dataset):
    def __init__(self, n: int = 1200, seed: int = 0):
        self.products = make_products(seed=seed)
        rng = random.Random(seed + 13)
        self.samples = []
        all_queries = [product_query(p) for p in self.products]
        self.vocab = Vocab(all_queries + [p.title for p in self.products])
        for _ in range(n):
            target = rng.choice(self.products)
            history = [rng.choice([p for p in self.products if p.category == target.category or p.intent == target.intent]) for _ in range(8)]
            negatives = rng.sample([q for q in all_queries if q != product_query(target)], 5)
            candidates = [product_query(target)] + negatives
            rng.shuffle(candidates)
            label = candidates.index(product_query(target))
            target_products = {p.pid for p in self.products if p.category == target.category and p.intent == target.intent}
            self.samples.append({"history": history, "candidates": candidates, "label": label, "target_products": target_products})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def collate(batch, vocab: Vocab):
    hist = []
    cand = []
    labels = []
    target_sets = []
    for b in batch:
        hist.append(torch.stack([vocab.encode(p.title) for p in b["history"]]))
        cand.append(torch.stack([vocab.encode(q) for q in b["candidates"]]))
        labels.append(b["label"])
        target_sets.append(b["target_products"])
    return {"history": torch.stack(hist), "candidates": torch.stack(cand), "labels": torch.tensor(labels), "target_sets": target_sets}
