from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

TOKEN_RE = re.compile(r"[a-zA-Z]+(?:-[a-zA-Z]+)?")


def tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]


class Vocabulary:
    def __init__(self, texts: Iterable[str], min_freq: int = 1) -> None:
        counts = Counter(token for text in texts for token in tokenize(text))
        self.itos = ["<pad>", "<unk>"] + sorted(t for t, c in counts.items() if c >= min_freq)
        self.stoi = {token: i for i, token in enumerate(self.itos)}

    def encode(self, text: str) -> torch.Tensor:
        ids = [self.stoi.get(token, 1) for token in tokenize(text)]
        if not ids:
            ids = [1]
        return torch.tensor(ids, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.itos)


class ProductEncoder(nn.Module):
    """Tiny CLIP-like encoder for toy image/text retrieval."""

    def __init__(self, vocab_size: int, num_products: int, dim: int = 128) -> None:
        super().__init__()
        self.text_embedding = nn.Embedding(vocab_size, dim)
        self.product_embedding = nn.Embedding(num_products, dim)
        self.text_projection = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
        self.image_projection = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))

    def encode_text_ids(self, token_ids: torch.Tensor) -> torch.Tensor:
        emb = self.text_embedding(token_ids)
        pooled = emb.mean(dim=0)
        return F.normalize(self.text_projection(pooled), dim=-1)

    def encode_product_ids(self, product_ids: torch.Tensor) -> torch.Tensor:
        emb = self.product_embedding(product_ids)
        return F.normalize(self.image_projection(emb), dim=-1)

    def forward(self, product_ids: torch.Tensor, padded_text: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        img = self.encode_product_ids(product_ids)
        txt_emb = self.text_embedding(padded_text)
        txt = (txt_emb * mask[..., None]).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        txt = F.normalize(self.text_projection(txt), dim=-1)
        return self.logit_scale.exp() * img @ txt.t()


def pad_text_batch(encoded: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(x.numel() for x in encoded)
    batch = torch.zeros(len(encoded), max_len, dtype=torch.long)
    mask = torch.zeros(len(encoded), max_len, dtype=torch.float32)
    for i, ids in enumerate(encoded):
        batch[i, : ids.numel()] = ids
        mask[i, : ids.numel()] = 1.0
    return batch, mask


@torch.no_grad()
def encode_titles(model: ProductEncoder, vocab: Vocabulary, titles: Sequence[str], device: str) -> torch.Tensor:
    vectors = [model.encode_text_ids(vocab.encode(title).to(device)) for title in titles]
    return torch.stack(vectors, dim=0)


@torch.no_grad()
def encode_products(model: ProductEncoder, product_ids: Sequence[int], device: str) -> torch.Tensor:
    ids = torch.tensor(product_ids, dtype=torch.long, device=device)
    return model.encode_product_ids(ids)


def captionfuse_query(image_vec: torch.Tensor, caption_vec: torch.Tensor, alpha: float = 0.15) -> torch.Tensor:
    return F.normalize(alpha * image_vec + (1.0 - alpha) * caption_vec, dim=-1)


def recall_at_k(ranks: Sequence[int], k: int) -> float:
    return float(np.mean([rank <= k for rank in ranks]))


def mean_average_precision(ranks: Sequence[int]) -> float:
    return float(np.mean([1.0 / rank for rank in ranks]))


def rank_targets(query_vectors: torch.Tensor, gallery_vectors: torch.Tensor, target_ids: Sequence[int]) -> List[int]:
    scores = query_vectors @ gallery_vectors.t()
    ranks: List[int] = []
    for i, target_id in enumerate(target_ids):
        order = torch.argsort(scores[i], descending=True)
        rank = (order == target_id).nonzero(as_tuple=False).item() + 1
        ranks.append(rank)
    return ranks


def summarize_metrics(ranks: Sequence[int]) -> Dict[str, float]:
    return {
        "r@1": recall_at_k(ranks, 1),
        "r@5": recall_at_k(ranks, 5),
        "r@10": recall_at_k(ranks, 10),
        "map": mean_average_precision(ranks),
        "median_rank": float(np.median(ranks)),
    }
