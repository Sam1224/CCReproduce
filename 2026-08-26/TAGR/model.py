from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TagrOutput:
    scores: torch.Tensor  # [B, C]
    live_token: torch.Tensor  # [B, D]
    intent: torch.Tensor  # [B, D]


class TAGRModel(nn.Module):
    def __init__(
        self,
        *,
        num_users: int,
        num_ads: int,
        num_rooms: int,
        num_categories: int,
        d: int = 32,
        hidden: int = 64,
    ) -> None:
        super().__init__()
        self.user = nn.Embedding(num_users, d)
        self.ad = nn.Embedding(num_ads, d)
        self.room = nn.Embedding(num_rooms, d)
        self.ad_cat = nn.Embedding(num_categories, d // 2)

        # Dynamic live-state token: room + stage + focused product category + promotion + freshness.
        self.stage = nn.Embedding(4, d // 2)
        self.focus_cat = nn.Embedding(num_categories, d // 2)
        self.promo = nn.Embedding(3, d // 2)
        self.live_mlp = nn.Sequential(nn.Linear(d + d * 2, hidden), nn.ReLU(), nn.Linear(hidden, d))

        # Intent-aware sequence encoder: action-aware history, queried by the current live token.
        self.action = nn.Embedding(4, d // 4)
        self.hist_proj = nn.Linear(d + d // 4, d)
        self.gru = nn.GRU(d, d, batch_first=True)
        self.query = nn.Linear(d, d)

        # Generative preference proxy: produce a user/live/intent-conditioned target vector.
        self.generator = nn.Sequential(nn.Linear(d * 3, hidden), nn.ReLU(), nn.Linear(hidden, d))
        self.scorer = nn.Sequential(nn.Linear(d + d // 2 + 2, hidden), nn.ReLU(), nn.Linear(hidden, d))
        self.bias = nn.Sequential(nn.Linear(d * 3, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    @torch.no_grad()
    def init_from_world(self, *, user_pref: torch.Tensor, ad_emb: torch.Tensor, room_style: torch.Tensor) -> None:
        self.user.weight.copy_(user_pref)
        self.ad.weight.copy_(ad_emb)
        self.room.weight.copy_(room_style)

    def forward(
        self,
        *,
        user_id: torch.Tensor,
        room_id: torch.Tensor,
        hist_item_ids: torch.Tensor,
        hist_actions: torch.Tensor,
        cand_item_ids: torch.Tensor,
        stage: torch.Tensor,
        focus_cat: torch.Tensor,
        promo: torch.Tensor,
        freshness: torch.Tensor,
        item_category: torch.Tensor,
        item_price: torch.Tensor,
        item_quality: torch.Tensor,
    ) -> TagrOutput:
        user = self.user(user_id)
        room = self.room(room_id)
        live_raw = torch.cat(
            [
                room,
                self.stage(stage),
                self.focus_cat(focus_cat),
                self.promo(promo),
                freshness[:, None].expand(-1, room.shape[-1] // 2),
            ],
            dim=-1,
        )
        live_token = torch.tanh(self.live_mlp(live_raw))

        hist = self.ad(hist_item_ids)
        hist = self.hist_proj(torch.cat([hist, self.action(hist_actions)], dim=-1))
        seq, _ = self.gru(hist)
        attn = torch.softmax((seq * self.query(live_token)[:, None, :]).sum(-1) / (seq.shape[-1] ** 0.5), dim=-1)
        intent = (seq * attn[:, :, None]).sum(1)

        pref = F.normalize(self.generator(torch.cat([user, live_token, intent], dim=-1)), dim=-1)
        cand = self.ad(cand_item_ids)
        side = torch.cat([self.ad_cat(item_category), item_price[..., None], item_quality[..., None]], dim=-1)
        cand_repr = F.normalize(self.scorer(torch.cat([cand, side], dim=-1)), dim=-1)
        logits = (cand_repr * pref[:, None, :]).sum(-1) * 8.0
        logits = logits + self.bias(torch.cat([user, live_token, intent], dim=-1))
        return TagrOutput(scores=logits, live_token=live_token, intent=intent)


def gather_item_side(world, cand_item_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        world.ad_category[cand_item_ids],
        world.ad_price[cand_item_ids],
        world.ad_quality[cand_item_ids],
    )


def tagr_loss(scores: torch.Tensor, labels: torch.Tensor, reward_proxy: torch.Tensor, alpha: float = 0.15) -> torch.Tensor:
    """Cross-entropy plus a lightweight online-preference proxy.

    The proxy rewards candidates that would have higher synthetic online value
    (conversion/revenue/freshness), mimicking intermittent online preference
    optimization without requiring a live serving loop.
    """
    ce = F.cross_entropy(scores, labels)
    policy = torch.softmax(scores, dim=-1)
    centered_reward = reward_proxy - reward_proxy.mean(dim=-1, keepdim=True)
    online_pref = -(policy * centered_reward.detach()).sum(dim=-1).mean()
    return ce + alpha * online_pref
