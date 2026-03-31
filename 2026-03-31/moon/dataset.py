from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch


@dataclass
class Product:
    category_id: int
    attr_id: int
    images_full: torch.Tensor  # (M, d_img)
    images_core: torch.Tensor  # (M, d_img)
    token_ids: torch.Tensor  # (L,)
    token_types: torch.Tensor  # (L,) 0=other, 1=cat, 2=attr


@dataclass
class PairExample:
    query_token_ids: torch.Tensor
    query_token_types: torch.Tensor
    product: Product


def _seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def make_synthetic_catalog(
    n_products: int = 5000,
    n_categories: int = 40,
    n_attrs: int = 60,
    d_latent: int = 32,
    d_img: int = 64,
    n_images: int = 3,
    seed: int = 0,
) -> Tuple[List[Product], dict]:
    """Create a synthetic catalog.

    Each product has a latent factor z. Category/attribute are derived from z.
    Images are z + noise; 'core crop' reduces background noise.
    """

    rng = _seed(seed)

    # Simple token vocab: reserve IDs for category and attribute tokens.
    # token_types ensures the MoE can route them.
    vocab_other = 800
    cat_base = vocab_other
    attr_base = vocab_other + n_categories

    products: List[Product] = []

    for _ in range(n_products):
        z = rng.normal(size=(d_latent,)).astype(np.float32)
        category_id = int(np.argmax(z[:n_categories]) % n_categories)
        attr_id = int(np.argmax(z[-n_attrs:]) % n_attrs)

        # multi-image
        full = []
        core = []
        for _m in range(n_images):
            bg = rng.normal(scale=0.7, size=(d_latent,)).astype(np.float32)
            full_vec = z + 0.6 * bg
            core_vec = z + 0.15 * bg  # "core" suppresses background

            # project to image embedding space
            scale = np.float32(1.0 / np.sqrt(d_latent))
            proj = rng.normal(size=(d_latent, d_img)).astype(np.float32) * scale
            full.append((full_vec @ proj).astype(np.float32))
            core.append((core_vec @ proj).astype(np.float32))

        images_full = torch.tensor(np.stack(full, axis=0))
        images_core = torch.tensor(np.stack(core, axis=0))

        # Title tokens: [cat_token, attr_token] + other tokens
        other_len = int(rng.integers(4, 10))
        other_tokens = rng.integers(0, vocab_other, size=(other_len,)).astype(np.int64)
        token_ids = np.concatenate(
            [
                np.array([cat_base + category_id, attr_base + attr_id], dtype=np.int64),
                other_tokens,
            ]
        )
        token_types = np.concatenate(
            [
                np.array([1, 2], dtype=np.int64),
                np.zeros((other_len,), dtype=np.int64),
            ]
        )

        products.append(
            Product(
                category_id=category_id,
                attr_id=attr_id,
                images_full=images_full,
                images_core=images_core,
                token_ids=torch.tensor(token_ids),
                token_types=torch.tensor(token_types),
            )
        )

    meta = {
        "vocab_size": vocab_other + n_categories + n_attrs,
        "n_categories": n_categories,
        "n_attrs": n_attrs,
        "cat_base": cat_base,
        "attr_base": attr_base,
        "d_img": d_img,
    }
    return products, meta


def make_pairs(products: List[Product], n_pairs: int = 6000, seed: int = 1) -> List[PairExample]:
    """Generate (query, product) pairs.

    We mimic 'purchase behavior supervision' by pairing a query derived from
    product aspects with that product.
    """

    rng = _seed(seed)
    pairs: List[PairExample] = []
    for _ in range(n_pairs):
        p = products[int(rng.integers(0, len(products)))]

        # Query tokens: mostly reuse the product's cat/attr, plus noise tokens.
        noise_len = int(rng.integers(2, 6))
        noise = rng.integers(0, 300, size=(noise_len,)).astype(np.int64)

        q_ids = torch.cat([p.token_ids[:2], torch.tensor(noise, dtype=torch.long)], dim=0)
        q_types = torch.cat([p.token_types[:2], torch.zeros((noise_len,), dtype=torch.long)], dim=0)
        pairs.append(PairExample(query_token_ids=q_ids, query_token_types=q_types, product=p))
    return pairs


def split(items: List, frac: float = 0.9) -> Tuple[List, List]:
    n = int(len(items) * frac)
    return items[:n], items[n:]
