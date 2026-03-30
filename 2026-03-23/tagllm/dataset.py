from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch


TAGS = [
    "fashion",
    "beauty",
    "food",
    "travel",
    "gaming",
    "fitness",
    "electronics",
    "home",
]

KEYWORDS: Dict[str, List[str]] = {
    "fashion": ["dress", "style", "outfit", "bag"],
    "beauty": ["skincare", "makeup", "serum", "lip"],
    "food": ["recipe", "spicy", "noodle", "coffee"],
    "travel": ["flight", "hotel", "beach", "city"],
    "gaming": ["rank", "fps", "rpg", "console"],
    "fitness": ["workout", "protein", "cardio", "yoga"],
    "electronics": ["phone", "laptop", "camera", "chip"],
    "home": ["sofa", "kitchen", "lamp", "clean"],
}


@dataclass
class Dataset:
    x: torch.Tensor  # (N, L)
    y: torch.Tensor  # (N, T)
    vocab: Dict[str, int]


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in text.replace(",", " ").replace(".", " ").split() if t]


def build_dataset(n: int = 2000, seed: int = 1, max_len: int = 24) -> Dataset:
    g = torch.Generator().manual_seed(seed)

    # Build synthetic notes.
    notes: List[List[str]] = []
    labels: List[List[int]] = []

    for _ in range(n):
        # sample 1-3 tags
        tcount = int(torch.randint(1, 4, (1,), generator=g).item())
        idx = torch.randperm(len(TAGS), generator=g)[:tcount].tolist()
        chosen = [TAGS[i] for i in idx]

        words = []
        for t in chosen:
            kw = KEYWORDS[t]
            # 2 keywords per tag
            for j in torch.randperm(len(kw), generator=g)[:2].tolist():
                words.append(kw[j])

        # add some noise tokens
        noise_pool = ["nice", "deal", "today", "review", "best", "new", "try"]
        for j in torch.randint(0, len(noise_pool), (4,), generator=g).tolist():
            words.append(noise_pool[j])

        # shuffle
        perm = torch.randperm(len(words), generator=g).tolist()
        words = [words[i] for i in perm]

        y = [0] * len(TAGS)
        for t in chosen:
            y[TAGS.index(t)] = 1

        notes.append(words)
        labels.append(y)

    # vocab
    vocab = {"[PAD]": 0, "[UNK]": 1}
    for wlist in notes:
        for w in wlist:
            if w not in vocab:
                vocab[w] = len(vocab)

    x = torch.zeros(n, max_len, dtype=torch.long)
    for i, wlist in enumerate(notes):
        ids = [vocab.get(w, 1) for w in wlist][:max_len]
        x[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)

    y = torch.tensor(labels, dtype=torch.float32)
    return Dataset(x=x, y=y, vocab=vocab)


def split(ds: Dataset, frac: float = 0.8) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    n = ds.x.shape[0]
    idx = torch.randperm(n)
    k = int(n * frac)
    tr = idx[:k]
    te = idx[k:]
    return (ds.x[tr], ds.y[tr]), (ds.x[te], ds.y[te])
