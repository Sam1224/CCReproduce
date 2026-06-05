from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Vocab:
    token_to_id: dict[str, int]
    id_to_token: list[str]

    @classmethod
    def from_catalog(cls, catalog_tokens: list[list[str]]) -> "Vocab":
        uniq = sorted({t for tokens in catalog_tokens for t in tokens})
        id_to_token = ["<pad>", "<unk>"] + uniq
        token_to_id = {t: i for i, t in enumerate(id_to_token)}
        return cls(token_to_id=token_to_id, id_to_token=id_to_token)

    def encode(self, tokens: list[str]) -> list[int]:
        unk = self.token_to_id["<unk>"]
        return [self.token_to_id.get(t, unk) for t in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        out: list[str] = []
        for i in ids:
            if 0 <= i < len(self.id_to_token):
                out.append(self.id_to_token[i])
            else:
                out.append("<unk>")
        return out


class ItemEncoder(nn.Module):
    def __init__(self, n_items: int, vocab_size: int, embed_dim: int, item_token_ids: torch.Tensor):
        super().__init__()
        self.item_embed = nn.Embedding(n_items, embed_dim)
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.item_token_ids = nn.Parameter(item_token_ids, requires_grad=False)

        nn.init.normal_(self.item_embed.weight, std=0.02)
        nn.init.normal_(self.token_embed.weight, std=0.02)

    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        # item_ids: [B]
        base = self.item_embed(item_ids)
        tok = self.item_token_ids[item_ids]  # [B, T]
        tok_emb = self.token_embed(tok).mean(dim=1)
        return base + tok_emb


class MemoryAbstraction(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        nn.init.normal_(self.token_embed.weight, std=0.02)

    def forward(self, history_token_ids: torch.Tensor) -> torch.Tensor:
        # history_token_ids: [B, H, T]
        return self.token_embed(history_token_ids).mean(dim=(1, 2))


class QueryPolicy(nn.Module):
    def __init__(self, embed_dim: int, vocab_size: int, max_query_len: int = 2):
        super().__init__()
        self.max_query_len = max_query_len
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList([nn.Linear(embed_dim, vocab_size) for _ in range(max_query_len)])
        self.value_head = nn.Linear(embed_dim, 1)

    def forward(self, user_state: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        # user_state: [B, D]
        h = self.mlp(user_state)
        logits = [head(h) for head in self.heads]
        value = self.value_head(h).squeeze(-1)
        return logits, value


class QueryAgentR1Toy(nn.Module):
    def __init__(
        self,
        *,
        n_items: int,
        vocab_size: int,
        embed_dim: int,
        item_token_ids: torch.Tensor,
        max_query_len: int = 2,
    ) -> None:
        super().__init__()
        self.item_encoder = ItemEncoder(n_items=n_items, vocab_size=vocab_size, embed_dim=embed_dim, item_token_ids=item_token_ids)
        self.memory = MemoryAbstraction(vocab_size=vocab_size, embed_dim=embed_dim)
        self.policy = QueryPolicy(embed_dim=embed_dim, vocab_size=vocab_size, max_query_len=max_query_len)

    def user_state(self, history_item_token_ids: torch.Tensor) -> torch.Tensor:
        return self.memory(history_item_token_ids)

    @torch.no_grad()
    def retrieve(self, query_token_ids: torch.Tensor, all_item_token_ids: torch.Tensor, topk: int = 10) -> torch.Tensor:
        # query_token_ids: [B, L]
        query_emb = self.item_encoder.token_embed(query_token_ids).mean(dim=1)  # [B, D]
        item_emb = self.item_encoder.forward(torch.arange(all_item_token_ids.shape[0], device=query_token_ids.device))
        scores = query_emb @ item_emb.T  # [B, N]
        return scores.topk(k=topk, dim=1).indices


def supervised_loss(logits: list[torch.Tensor], target_token_ids: torch.Tensor) -> torch.Tensor:
    # target_token_ids: [B, L]
    losses = []
    for i, logit in enumerate(logits):
        losses.append(F.cross_entropy(logit, target_token_ids[:, i]))
    return sum(losses) / len(losses)


def sample_actions(logits: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    # returns (actions [B,L], log_prob [B])
    actions = []
    logps = []
    for logit in logits:
        dist = torch.distributions.Categorical(logits=logit)
        a = dist.sample()
        actions.append(a)
        logps.append(dist.log_prob(a))
    act = torch.stack(actions, dim=1)
    logp = torch.stack(logps, dim=1).sum(dim=1)
    return act, logp
