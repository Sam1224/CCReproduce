from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F

from data import (
    BOS,
    EOS,
    FILLER_OFFSET,
    FILLER_TOKEN_IDS,
    PAD,
    STYLE_OFFSET,
    STYLE_TOKEN_IDS,
    TOPIC_OFFSET,
    TOPIC_TOKEN_IDS,
    VOCAB_SIZE,
)


@dataclass
class SampleOutput:
    titles: torch.Tensor  # [B, K, L]
    log_probs: torch.Tensor  # [B, K]


class Generator(nn.Module):
    """Toy conditional headline generator (exploration).

    To keep the reproduction fast and runnable on CPU, we model the headline as 3 conditional
    categorical decisions:

        headline = [BOS, STYLE, TOPIC, FILLER, EOS]

    This still captures the *shape* of "generate multiple candidates" in GESE.
    """

    def __init__(self, n_users: int, n_topics: int, hidden_dim: int = 72, emb_dim: int = 72):
        super().__init__()
        self.n_users = n_users
        self.n_topics = n_topics

        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.topic_emb = nn.Embedding(n_topics, emb_dim)

        self.ctx = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.style_head = nn.Linear(hidden_dim, len(STYLE_TOKEN_IDS))
        self.topic_head = nn.Linear(hidden_dim, len(TOPIC_TOKEN_IDS))
        self.filler_head = nn.Linear(hidden_dim, len(FILLER_TOKEN_IDS))

    def _ctx(self, user_id: torch.Tensor, item_topic: torch.Tensor) -> torch.Tensor:
        u = self.user_emb(user_id)
        it = self.topic_emb(item_topic)
        return self.ctx(torch.cat([u, it], dim=-1))

    @torch.no_grad()
    def sample(
        self,
        user_id: torch.Tensor,
        item_topic: torch.Tensor,
        num_candidates: int = 8,
        temperature: float = 1.2,
    ) -> SampleOutput:
        device = user_id.device
        bsz = int(user_id.shape[0])

        user = user_id.repeat_interleave(num_candidates)
        topic = item_topic.repeat_interleave(num_candidates)

        ctx = self._ctx(user, topic)

        style_logits = self.style_head(ctx) / max(1e-6, temperature)
        topic_logits = self.topic_head(ctx) / max(1e-6, temperature)
        filler_logits = self.filler_head(ctx) / max(1e-6, temperature)

        style_prob = F.softmax(style_logits, dim=-1)
        topic_prob = F.softmax(topic_logits, dim=-1)
        filler_prob = F.softmax(filler_logits, dim=-1)

        style_idx = torch.multinomial(style_prob, num_samples=1).squeeze(1)
        topic_idx = torch.multinomial(topic_prob, num_samples=1).squeeze(1)
        filler_idx = torch.multinomial(filler_prob, num_samples=1).squeeze(1)

        logp = (
            torch.log(style_prob.gather(1, style_idx.unsqueeze(1)).squeeze(1) + 1e-12)
            + torch.log(topic_prob.gather(1, topic_idx.unsqueeze(1)).squeeze(1) + 1e-12)
            + torch.log(filler_prob.gather(1, filler_idx.unsqueeze(1)).squeeze(1) + 1e-12)
        )

        titles = torch.full((user.shape[0], 5), PAD, dtype=torch.long, device=device)
        titles[:, 0] = BOS
        titles[:, 1] = STYLE_OFFSET + style_idx
        titles[:, 2] = TOPIC_OFFSET + topic_idx
        titles[:, 3] = FILLER_OFFSET + filler_idx
        titles[:, 4] = EOS

        titles = titles.view(bsz, num_candidates, -1)
        logp = logp.view(bsz, num_candidates)
        return SampleOutput(titles=titles, log_probs=logp)

    def log_prob(self, user_id: torch.Tensor, item_topic: torch.Tensor, titles: torch.Tensor) -> torch.Tensor:
        """Sum log-prob for (style, topic, filler).

        titles: [N,5]
        returns: [N]
        """

        ctx = self._ctx(user_id, item_topic)
        style_lp = F.log_softmax(self.style_head(ctx), dim=-1)
        topic_lp = F.log_softmax(self.topic_head(ctx), dim=-1)
        filler_lp = F.log_softmax(self.filler_head(ctx), dim=-1)

        style_idx = (titles[:, 1] - STYLE_OFFSET).clamp(min=0, max=len(STYLE_TOKEN_IDS) - 1)
        topic_idx = (titles[:, 2] - TOPIC_OFFSET).clamp(min=0, max=len(TOPIC_TOKEN_IDS) - 1)
        filler_idx = (titles[:, 3] - FILLER_OFFSET).clamp(min=0, max=len(FILLER_TOKEN_IDS) - 1)

        return (
            style_lp.gather(1, style_idx.unsqueeze(1)).squeeze(1)
            + topic_lp.gather(1, topic_idx.unsqueeze(1)).squeeze(1)
            + filler_lp.gather(1, filler_idx.unsqueeze(1)).squeeze(1)
        )


class TitleEncoder(nn.Module):
    def __init__(self, emb_dim: int = 64):
        super().__init__()
        self.token_emb = nn.Embedding(VOCAB_SIZE, emb_dim, padding_idx=PAD)
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, title_tokens: torch.Tensor) -> torch.Tensor:
        x = self.token_emb(title_tokens)  # [B,L,E]
        pooled = x.mean(dim=1)
        return self.proj(pooled)


class Selector(nn.Module):
    """A lightweight exploitation model scoring (user,item,title)."""

    def __init__(self, n_users: int, n_topics: int, hidden_dim: int = 128, emb_dim: int = 72):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.topic_emb = nn.Embedding(n_topics, emb_dim)
        self.title_enc = TitleEncoder(emb_dim=emb_dim)

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, user_id: torch.Tensor, item_topic: torch.Tensor, title_tokens: torch.Tensor) -> torch.Tensor:
        u = self.user_emb(user_id)
        it = self.topic_emb(item_topic)
        t = self.title_enc(title_tokens)
        score = self.mlp(torch.cat([u, it, t], dim=-1)).squeeze(-1)
        return torch.sigmoid(score)


def pick_best_by_logprob(sample_out: SampleOutput) -> Tuple[torch.Tensor, torch.Tensor]:
    idx = sample_out.log_probs.argmax(dim=1)
    bsz, _, l = sample_out.titles.shape
    gather = idx.view(bsz, 1, 1).expand(bsz, 1, l)
    best = sample_out.titles.gather(1, gather).squeeze(1)
    return best, idx


@torch.no_grad()
def pick_best_by_selector(
    selector: Selector, sample_out: SampleOutput, user_id: torch.Tensor, item_topic: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    bsz, k, l = sample_out.titles.shape
    flat_titles = sample_out.titles.view(bsz * k, l)
    flat_user = user_id.repeat_interleave(k)
    flat_topic = item_topic.repeat_interleave(k)
    scores = selector(flat_user, flat_topic, flat_titles).view(bsz, k)
    idx = scores.argmax(dim=1)
    gather = idx.view(bsz, 1, 1).expand(bsz, 1, l)
    best = sample_out.titles.gather(1, gather).squeeze(1)
    return best, idx
