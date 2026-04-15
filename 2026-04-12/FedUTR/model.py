from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch
from torch import nn


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, text_dim: int):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, text_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: [B, T]
        emb = self.token_emb(token_ids)  # [B, T, D]
        return emb.mean(dim=1)


class FedUTRShared(nn.Module):
    """Shared (server-aggregated) part of FedUTR."""

    def __init__(
        self,
        *,
        vocab_size: int,
        item_count: int,
        dim: int = 64,
        text_dim: int = 64,
    ):
        super().__init__()
        self.item_emb = nn.Embedding(item_count, dim)
        self.text_encoder = TextEncoder(vocab_size=vocab_size, text_dim=text_dim)
        self.urm = nn.Linear(text_dim, dim)

        # CIFM (simplified): gate between ID embedding and universal text embedding.
        self.cifm_gate = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid(),
        )

    def universal_item_embedding(self, item_token_ids: torch.Tensor) -> torch.Tensor:
        # [B, T] -> [B, D]
        return self.urm(self.text_encoder(item_token_ids))

    def fused_item_embedding(
        self,
        *,
        user_vec: torch.Tensor,  # [B, D]
        item_id: torch.Tensor,  # [B]
        item_token_ids: torch.Tensor,  # [B, T]
    ) -> torch.Tensor:
        id_vec = self.item_emb(item_id)
        uni_vec = self.universal_item_embedding(item_token_ids)
        gate_in = torch.cat([user_vec, id_vec, uni_vec], dim=-1)
        gate = self.cifm_gate(gate_in)  # [B, 1]
        return gate * id_vec + (1 - gate) * uni_vec


@dataclass
class ClientLocal:
    user_emb: nn.Embedding
    user_bias: nn.Parameter

    def parameters(self):
        return list(self.user_emb.parameters()) + [self.user_bias]


def init_clients(*, clients: int, users_per_client: int, dim: int, device: torch.device) -> Dict[int, ClientLocal]:
    locals_: Dict[int, ClientLocal] = {}
    for cid in range(clients):
        user_emb = nn.Embedding(users_per_client, dim).to(device)
        user_bias = nn.Parameter(torch.zeros(dim, device=device))
        locals_[cid] = ClientLocal(user_emb=user_emb, user_bias=user_bias)
    return locals_


def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    # -log sigmoid(pos - neg)
    return -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-12).mean()
