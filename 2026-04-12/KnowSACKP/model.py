from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from torch import nn


class TokenMeanEncoder(nn.Module):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.emb(token_ids).mean(dim=1)


@dataclass
class MockLLM:
    """A toy 'LLM ranker' with item-level knowledge gaps.

    - Popular items: parametric representation is close to ground truth latent.
    - Tail items: parametric representation is noisy.

    External augmentation provides a more stable 'attribute embedding'.
    """

    item_latent: np.ndarray  # [I, D]
    item_popularity: np.ndarray  # [I]
    vocab_size: int
    item_tokens: List[List[int]]

    def build_modules(self, device: torch.device, dim: int = 32):
        self.device = device
        self.dim = dim
        self.attr_encoder = TokenMeanEncoder(self.vocab_size, dim).to(device)

        latent = torch.tensor(self.item_latent[:, :dim], dtype=torch.float32, device=device)
        pop = torch.tensor(self.item_popularity, dtype=torch.float32, device=device)

        # Parametric embeddings: blend latent with popularity-dependent noise.
        noise = torch.randn_like(latent) * (1.2 - pop.unsqueeze(1))
        self.param_emb = (0.7 + 0.3 * pop.unsqueeze(1)) * latent + 0.3 * noise

        # External attribute embeddings (stable): assume retrieved metadata is strongly correlated
        # with the true item semantics, but still derived from tokens for interface consistency.
        token_tensor = self._item_token_tensor()
        with torch.no_grad():
            token_feat = self.attr_encoder(token_tensor)
            # External knowledge is more helpful for tail items, while for popular items it can be
            # redundant/noisy (too much context, conflicting descriptions, etc.).
            attr_noise = torch.randn_like(latent) * (0.6 * pop.unsqueeze(1))
            self.attr_emb = 0.9 * latent + 0.1 * token_feat + 0.2 * attr_noise

    def _item_token_tensor(self) -> torch.Tensor:
        max_len = max(len(t) for t in self.item_tokens)
        padded = [t + [0] * (max_len - len(t)) for t in self.item_tokens]
        return torch.tensor(padded, dtype=torch.long, device=self.device)

    def score_items(self, user_vec: torch.Tensor, item_ids: torch.Tensor, item_repr: torch.Tensor) -> torch.Tensor:
        # Simple dot product.
        return (user_vec * item_repr[item_ids]).sum(dim=-1)

    def get_item_repr(self, policy: str, item_ids: torch.Tensor, augment_mask: torch.Tensor | None = None) -> torch.Tensor:
        if policy == "no_augment":
            return self.param_emb
        if policy == "uniform":
            # Uniform augmentation may introduce strong distraction for well-known items:
            # extra context is redundant and can blur the ranking signal.
            pop = torch.tensor(self.item_popularity, dtype=torch.float32, device=self.device)
            mix = 0.9 - 0.4 * pop  # popular items rely more on parametric knowledge
            distraction = torch.randn_like(self.param_emb) * (2.0 * pop.unsqueeze(1))
            return (
                mix.unsqueeze(1) * self.attr_emb
                + (1 - mix).unsqueeze(1) * self.param_emb
                + distraction
            )
        if policy == "selective":
            if augment_mask is None:
                raise ValueError("augment_mask required for selective")
            out = self.param_emb.clone()
            out[augment_mask] = self.attr_emb[augment_mask]
            return out
        raise ValueError(policy)
