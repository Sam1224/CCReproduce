from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Product:
    product_id: str
    title: str
    attributes: Dict[str, str]


class SimpleTokenizer:
    """A tiny whitespace tokenizer for toy text + attribute sequences."""

    def __init__(self, vocab: Dict[str, int]):
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}

    @staticmethod
    def build(texts: Sequence[str], extra_tokens: Sequence[str] | None = None) -> "SimpleTokenizer":
        tokens: List[str] = []
        for text in texts:
            tokens.extend(text.lower().split())

        base = ["<pad>", "<bos>", "<eos>", "<unk>"]
        if extra_tokens:
            base.extend(list(extra_tokens))

        uniq = sorted(set(tokens))
        vocab = {t: i for i, t in enumerate(base + uniq)}
        return SimpleTokenizer(vocab)

    @property
    def pad_id(self) -> int:
        return self.vocab["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.vocab["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.vocab["<eos>"]

    @property
    def unk_id(self) -> int:
        return self.vocab["<unk>"]

    def encode(self, text: str, max_len: int) -> torch.Tensor:
        ids = [self.bos_id]
        for token in text.lower().split():
            ids.append(self.vocab.get(token, self.unk_id))
        ids.append(self.eos_id)

        ids = ids[:max_len]
        ids = ids + [self.pad_id] * (max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids: Sequence[int]) -> str:
        toks = []
        for idx in ids:
            tok = self.inv_vocab.get(int(idx), "<unk>")
            if tok in {"<pad>", "<bos>"}:
                continue
            if tok == "<eos>":
                break
            toks.append(tok)
        return " ".join(toks)


class ToyProductCatalog:
    """A deterministic synthetic product catalog.

    Each product has a title (text) and structured attributes.
    Images are not stored; we generate a deterministic tensor per product_id.
    """

    def __init__(self, seed: int = 13):
        self.seed = seed
        self.products: List[Product] = self._build_products()
        self.by_id = {p.product_id: p for p in self.products}

    def _build_products(self) -> List[Product]:
        rng = random.Random(self.seed)

        categories = ["dress", "sneaker", "handbag", "jacket", "watch"]
        colors = ["red", "blue", "black", "white", "green"]
        materials = ["leather", "cotton", "denim", "silk"]
        styles = ["casual", "formal", "sporty"]

        products: List[Product] = []
        idx = 0
        for category in categories:
            for color in colors:
                for material in materials:
                    style = rng.choice(styles)
                    title = f"{color} {material} {category} {style}"
                    product_id = f"p{idx:04d}"
                    products.append(
                        Product(
                            product_id=product_id,
                            title=title,
                            attributes={
                                "category": category,
                                "color": color,
                                "material": material,
                                "style": style,
                            },
                        )
                    )
                    idx += 1

        rng.shuffle(products)
        return products

    def image_tensor(self, product_id: str, size: int = 64) -> torch.Tensor:
        # Deterministic pseudo-image from product_id + seed.
        h = abs(hash((self.seed, product_id))) % (2**32)
        rng = np.random.default_rng(h)
        arr = rng.random((3, size, size), dtype=np.float32)

        # Add a simple structured pattern to mimic local details.
        yy, xx = np.mgrid[0:size, 0:size]
        arr[0] = (arr[0] * 0.7 + (xx / (size - 1)) * 0.3).astype(np.float32)
        arr[1] = (arr[1] * 0.7 + (yy / (size - 1)) * 0.3).astype(np.float32)
        return torch.tensor(arr, dtype=torch.float32)


class ToyMOONTripletDataset(Dataset):
    """Build (query, positive, negative) triplets from the product catalog.

    Positive: same category + color, different material if possible.
    Negative: different category.

    We also expose a target attribute sequence to supervise the attribute deconstruction head.
    """

    def __init__(
        self,
        catalog: ToyProductCatalog,
        num_samples: int = 5000,
        seed: int = 7,
        image_size: int = 64,
    ):
        self.catalog = catalog
        self.num_samples = num_samples
        self.seed = seed
        self.image_size = image_size

        self._rng = random.Random(seed)
        self._triplets = [self._sample_triplet() for _ in range(num_samples)]

    def _sample_triplet(self) -> Tuple[str, str, str]:
        q = self._rng.choice(self.catalog.products)

        positives = [
            p
            for p in self.catalog.products
            if p.product_id != q.product_id
            and p.attributes["category"] == q.attributes["category"]
            and p.attributes["color"] == q.attributes["color"]
        ]
        if not positives:
            positives = [p for p in self.catalog.products if p.product_id != q.product_id]
        p = self._rng.choice(positives)

        negatives = [p for p in self.catalog.products if p.attributes["category"] != q.attributes["category"]]
        n = self._rng.choice(negatives)

        return q.product_id, p.product_id, n.product_id

    @staticmethod
    def attributes_to_text(attributes: Dict[str, str]) -> str:
        # A compact structured "reasoning" target.
        # In the real paper this is a CoT-like attribute deconstruction.
        return " ".join([
            f"category:{attributes['category']}",
            f"color:{attributes['color']}",
            f"material:{attributes['material']}",
            f"style:{attributes['style']}",
        ])

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Dict[str, object]:
        qid, pid, nid = self._triplets[index]
        q = self.catalog.by_id[qid]
        p = self.catalog.by_id[pid]
        n = self.catalog.by_id[nid]

        return {
            "q_id": qid,
            "p_id": pid,
            "n_id": nid,
            "q_image": self.catalog.image_tensor(qid, size=self.image_size),
            "p_image": self.catalog.image_tensor(pid, size=self.image_size),
            "n_image": self.catalog.image_tensor(nid, size=self.image_size),
            "q_text": q.title,
            "p_text": p.title,
            "n_text": n.title,
            "q_attr_text": self.attributes_to_text(q.attributes),
        }


def collate_triplets(
    batch: Sequence[Dict[str, object]],
    text_tokenizer: SimpleTokenizer,
    attr_tokenizer: SimpleTokenizer,
    max_text_len: int = 16,
    max_attr_len: int = 20,
) -> Dict[str, torch.Tensor]:
    def stack_images(key: str) -> torch.Tensor:
        return torch.stack([b[key] for b in batch])  # type: ignore[arg-type]

    q_text = torch.stack([text_tokenizer.encode(b["q_text"], max_text_len) for b in batch])
    p_text = torch.stack([text_tokenizer.encode(b["p_text"], max_text_len) for b in batch])
    n_text = torch.stack([text_tokenizer.encode(b["n_text"], max_text_len) for b in batch])

    # Attribute sequence supervision (teacher forcing).
    q_attr = torch.stack([attr_tokenizer.encode(b["q_attr_text"], max_attr_len) for b in batch])

    return {
        "q_image": stack_images("q_image"),
        "p_image": stack_images("p_image"),
        "n_image": stack_images("n_image"),
        "q_text": q_text,
        "p_text": p_text,
        "n_text": n_text,
        "q_attr": q_attr,
    }
