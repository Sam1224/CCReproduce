from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


_WORD_RE = re.compile(r"[a-z0-9]+")


def simple_tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


@dataclass(frozen=True)
class FashionItem:
    category: str
    color: str
    material: str
    pattern: str
    style: str


ATTRIBUTE_SETS = {
    "category": ["dress", "shirt", "pants", "shoes", "bag"],
    "color": ["red", "blue", "green", "black", "white"],
    "material": ["cotton", "denim", "leather", "silk", "wool"],
    "pattern": ["solid", "striped", "polka", "plaid", "floral"],
    "style": ["casual", "formal", "sport", "vintage", "street"],
}


def item_to_short_query(item: FashionItem) -> str:
    return f"{item.color} {item.material} {item.category}"


def item_to_long_query(item: FashionItem) -> str:
    # Long query mimics verbose user intent commonly seen in e-commerce search.
    return (
        "i am looking for a "
        f"{item.style} {item.pattern} {item.color} {item.material} {item.category} "
        "that matches my outfit for the weekend and feels comfortable"
    )


def _color_rgb(name: str) -> tuple[float, float, float]:
    table = {
        "red": (0.85, 0.15, 0.15),
        "blue": (0.18, 0.34, 0.85),
        "green": (0.16, 0.70, 0.30),
        "black": (0.08, 0.08, 0.10),
        "white": (0.92, 0.92, 0.92),
    }
    return table.get(name, (0.5, 0.5, 0.5))


def render_item_image(item: FashionItem, size: int = 64) -> torch.Tensor:
    """Render a deterministic toy 'product image' tensor.

    The goal is NOT photo realism. We only need a stable mapping from structured
    attributes -> pixels so that a CNN can learn retrieval.

    Returns: float tensor in [0, 1], shape (3, H, W)
    """

    r, g, b = _color_rgb(item.color)
    img = torch.zeros(3, size, size)
    img[0].fill_(r)
    img[1].fill_(g)
    img[2].fill_(b)

    # Pattern
    if item.pattern == "striped":
        img[:, :, ::6] = 1.0
    elif item.pattern == "polka":
        yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
        mask = ((xx - size / 2) ** 2 + (yy - size / 2) ** 2) % 240 < 60
        img[:, mask] = 1.0
    elif item.pattern == "plaid":
        img[:, ::6, :] = 1.0
        img[:, :, ::6] = 1.0
    elif item.pattern == "floral":
        yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
        petals = (torch.sin(xx / 4.0) * torch.cos(yy / 5.0)).abs()
        img = img * (0.6 + 0.4 * petals)

    # Category silhouette
    if item.category in {"shoes", "bag"}:
        img[:, : size // 3, :] *= 0.85
    elif item.category == "dress":
        img[:, :, : size // 4] *= 0.85
        img[:, :, -size // 4 :] *= 0.85

    # Material texture
    if item.material == "denim":
        img = img * 0.9 + 0.1 * torch.rand_like(img) * 0.3
    elif item.material == "leather":
        img = img * 0.85
    elif item.material == "silk":
        yy = torch.arange(size).float().view(1, -1, 1)
        sheen = (torch.sin(yy / 6.0) + 1.0) / 2.0
        img = img * (0.8 + 0.2 * sheen)

    return img.clamp(0.0, 1.0)


class Vocab:
    def __init__(self, tokens: Iterable[str]):
        uniq = ["[pad]", "[unk]"]
        seen = set(uniq)
        for t in tokens:
            if t not in seen:
                uniq.append(t)
                seen.add(t)
        self.stoi = {t: i for i, t in enumerate(uniq)}
        self.itos = uniq

    def encode(self, text: str, max_len: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
        toks = simple_tokenize(text)
        ids = [self.stoi.get(t, self.stoi["[unk]"]) for t in toks][:max_len]
        if len(ids) < max_len:
            ids += [self.stoi["[pad]"]] * (max_len - len(ids))
        attn = [0 if i == self.stoi["[pad]"] else 1 for i in ids]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(attn, dtype=torch.long)


def build_default_vocab() -> Vocab:
    tokens = []
    for vs in ATTRIBUTE_SETS.values():
        tokens.extend(vs)
    tokens += [
        "i",
        "am",
        "looking",
        "for",
        "a",
        "that",
        "matches",
        "my",
        "outfit",
        "weekend",
        "feels",
        "comfortable",
        "and",
        "the",
        "with",
    ]
    return Vocab(tokens)


class TeacherEmbedder:
    """A deterministic 'teacher' that maps attributes to a continuous embedding."""

    def __init__(self, dim: int = 128, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.dim = dim
        self.attr_vec = {}
        for key, vals in ATTRIBUTE_SETS.items():
            for v in vals:
                self.attr_vec[(key, v)] = rng.standard_normal(dim).astype(np.float32)

    def embed_item(self, item: FashionItem) -> torch.Tensor:
        vec = (
            self.attr_vec[("category", item.category)]
            + self.attr_vec[("color", item.color)]
            + self.attr_vec[("material", item.material)]
            + self.attr_vec[("pattern", item.pattern)]
            + self.attr_vec[("style", item.style)]
        )
        v = torch.from_numpy(vec)
        return v / (v.norm(p=2) + 1e-8)

    def embed_text(self, text: str) -> torch.Tensor:
        toks = simple_tokenize(text)
        vec = np.zeros(self.dim, dtype=np.float32)
        for key, vals in ATTRIBUTE_SETS.items():
            for v in vals:
                if v in toks:
                    vec += self.attr_vec[(key, v)]
        if float(np.linalg.norm(vec)) < 1e-8:
            vec[0] = 1.0
        v = torch.from_numpy(vec)
        return v / (v.norm(p=2) + 1e-8)


class ToyFashionRetrievalDataset(Dataset):
    """Toy dataset for image<->text retrieval.

    Each sample is (image, short_query, long_query, item).

    - In-domain split uses the same attribute sets.
    - OOD split simulates distribution shift by resampling attribute priors.
    """

    def __init__(
        self,
        num_items: int = 800,
        image_size: int = 64,
        seed: int = 42,
        ood: bool = False,
    ):
        self.image_size = image_size
        self.teacher = TeacherEmbedder(dim=128, seed=0)

        rng = random.Random(seed)
        items: list[FashionItem] = []
        for _ in range(num_items):
            # OOD shift: skew distributions (e.g., unseen combos become more frequent).
            if ood:
                category = rng.choice(["dress", "bag", "bag", "bag", "shoes"])
                color = rng.choice(["black", "black", "white", "blue", "green"])
                material = rng.choice(["leather", "leather", "denim", "cotton", "wool"])
                pattern = rng.choice(["solid", "solid", "plaid", "striped", "floral"])
                style = rng.choice(["formal", "street", "street", "vintage", "casual"])
            else:
                category = rng.choice(ATTRIBUTE_SETS["category"])
                color = rng.choice(ATTRIBUTE_SETS["color"])
                material = rng.choice(ATTRIBUTE_SETS["material"])
                pattern = rng.choice(ATTRIBUTE_SETS["pattern"])
                style = rng.choice(ATTRIBUTE_SETS["style"])

            items.append(FashionItem(category, color, material, pattern, style))

        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        img = render_item_image(item, size=self.image_size)
        short = item_to_short_query(item)
        long = item_to_long_query(item)

        # teacher embeddings used for distillation
        t_img = self.teacher.embed_item(item)
        t_short = self.teacher.embed_text(short)
        t_long = self.teacher.embed_text(long)

        return {
            "image": img,
            "short_query": short,
            "long_query": long,
            "teacher_image_emb": t_img,
            "teacher_short_emb": t_short,
            "teacher_long_emb": t_long,
        }


def collate(batch: list[dict], vocab: Vocab, max_len: int = 32) -> dict:
    images = torch.stack([b["image"] for b in batch], dim=0)

    short_ids, short_attn = zip(*(vocab.encode(b["short_query"], max_len=max_len) for b in batch))
    long_ids, long_attn = zip(*(vocab.encode(b["long_query"], max_len=max_len) for b in batch))

    out = {
        "images": images,
        "short_ids": torch.stack(short_ids, dim=0),
        "short_attn": torch.stack(short_attn, dim=0),
        "long_ids": torch.stack(long_ids, dim=0),
        "long_attn": torch.stack(long_attn, dim=0),
        "teacher_image_emb": torch.stack([b["teacher_image_emb"] for b in batch], dim=0),
        "teacher_short_emb": torch.stack([b["teacher_short_emb"] for b in batch], dim=0),
        "teacher_long_emb": torch.stack([b["teacher_long_emb"] for b in batch], dim=0),
    }
    return out


def split_indices(n: int, train_ratio: float = 0.85, seed: int = 0) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    cut = int(math.floor(n * train_ratio))
    return idx[:cut], idx[cut:]
