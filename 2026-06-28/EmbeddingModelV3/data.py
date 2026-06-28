from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Product:
    pid: str
    title: str
    attrs: Dict[str, str]


@dataclass(frozen=True)
class Query:
    qid: str
    text: str
    intent: Dict[str, str]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


_BRANDS = ["acme", "freshco", "greenfarm", "snacklab", "vita", "yummy"]
_CATS = ["coffee", "tea", "snacks", "cereal", "sauce", "noodles"]
_FLAVORS = ["vanilla", "chocolate", "spicy", "sweet", "original", "lemon"]
_SIZES = ["small", "medium", "large"]
_DIET = ["regular", "gluten_free", "lactose_free", "vegan"]


def _tok(s: str) -> List[str]:
    return [t for t in s.lower().replace("-", " ").split() if t]


def build_vocab(texts: Iterable[str], min_freq: int = 1) -> Dict[str, int]:
    freq: Dict[str, int] = {}
    for t in texts:
        for w in _tok(t):
            freq[w] = freq.get(w, 0) + 1
    vocab = {"<pad>": 0, "<unk>": 1}
    for w, c in sorted(freq.items(), key=lambda x: (-x[1], x[0])):
        if c >= min_freq and w not in vocab:
            vocab[w] = len(vocab)
    return vocab


def encode(text: str, vocab: Dict[str, int]) -> List[int]:
    ids = []
    for w in _tok(text):
        ids.append(vocab.get(w, vocab["<unk>"]))
    return ids or [vocab["<unk>"]]


def make_toy_catalog(n_products: int, seed: int = 0) -> List[Product]:
    set_seed(seed)
    prods: List[Product] = []
    for i in range(n_products):
        brand = random.choice(_BRANDS)
        cat = random.choice(_CATS)
        flavor = random.choice(_FLAVORS)
        size = random.choice(_SIZES)
        diet = random.choice(_DIET)
        title = f"{brand} {cat} {flavor} {size} {diet}".replace("_", " ")
        attrs = {"brand": brand, "cat": cat, "flavor": flavor, "size": size, "diet": diet}
        prods.append(Product(pid=f"p{i:05d}", title=title, attrs=attrs))
    return prods


def make_toy_queries(n_queries: int, seed: int = 0) -> List[Query]:
    set_seed(seed)
    qs: List[Query] = []
    for i in range(n_queries):
        brand = random.choice(_BRANDS)
        cat = random.choice(_CATS)

        # flavor/size/diet are optional intents; this creates tail + ambiguity.
        intent: Dict[str, str] = {"brand": brand, "cat": cat}
        if random.random() < 0.55:
            intent["flavor"] = random.choice(_FLAVORS)
        if random.random() < 0.35:
            intent["diet"] = random.choice(_DIET)
        if random.random() < 0.20:
            intent["size"] = random.choice(_SIZES)

        parts = [brand, cat]
        for k in ["flavor", "diet", "size"]:
            if k in intent:
                parts.append(intent[k].replace("_", " "))
        text = " ".join(parts)
        qs.append(Query(qid=f"q{i:05d}", text=text, intent=intent))
    return qs


def relevance_grade(q: Query, p: Product) -> int:
    """5-grade relevance label in [0,4].

    This is the *ground truth* used for evaluation in the toy setting.
    """

    score = 0
    for k, v in q.intent.items():
        if p.attrs.get(k) == v:
            score += 1

    # brand+cat are mandatory intents, so grades are shaped around them.
    if p.attrs.get("brand") != q.intent.get("brand") or p.attrs.get("cat") != q.intent.get("cat"):
        return 0

    # start from 2 if brand+cat match.
    grade = 2
    extra = max(0, score - 2)
    grade = min(4, grade + extra)

    # penalize subtle mismatch for optional constraints
    if "diet" in q.intent and p.attrs.get("diet") != q.intent["diet"]:
        grade = max(1, grade - 1)
    return int(grade)


def train_valid_test_split(items: Sequence, ratios=(0.8, 0.1, 0.1), seed: int = 0):
    assert math.isclose(sum(ratios), 1.0)
    idx = list(range(len(items)))
    random.Random(seed).shuffle(idx)
    n = len(items)
    n_train = int(n * ratios[0])
    n_valid = int(n * ratios[1])
    train = [items[i] for i in idx[:n_train]]
    valid = [items[i] for i in idx[n_train : n_train + n_valid]]
    test = [items[i] for i in idx[n_train + n_valid :]]
    return train, valid, test


def build_corpus(queries: Sequence[Query], products: Sequence[Product]) -> Tuple[List[str], List[str]]:
    q_texts = [q.text for q in queries]
    p_texts = [p.title for p in products]
    return q_texts, p_texts
