from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from cq_sid.utils.tokenizer import Vocab, basic_tokenize


@dataclass(frozen=True)
class ToyItem:
    item_id: str
    category_id: int
    title: str


@dataclass(frozen=True)
class ToyUser:
    user_id: str
    profile: str


@dataclass(frozen=True)
class ToyExample:
    query: str
    user_profile: str
    item_id: str
    category_id: int


def build_toy_corpus() -> tuple[list[ToyItem], list[ToyUser], list[ToyExample]]:
    items = [
        ToyItem("i_001", 0, "hydrating facial moisturizer for sensitive skin"),
        ToyItem("i_002", 0, "anti aging retinol night cream"),
        ToyItem("i_003", 0, "vitamin c brightening serum"),
        ToyItem("i_010", 1, "wireless noise cancelling headphones"),
        ToyItem("i_011", 1, "bluetooth speaker waterproof"),
        ToyItem("i_012", 1, "mechanical keyboard compact"),
        ToyItem("i_020", 2, "running shoes breathable"),
        ToyItem("i_021", 2, "yoga mat non slip"),
        ToyItem("i_022", 2, "compression socks for travel"),
    ]

    users = [
        ToyUser("u_001", "prefers fragrance free skincare"),
        ToyUser("u_002", "buys electronics for gaming"),
        ToyUser("u_003", "runs marathons and does yoga"),
    ]

    examples = [
        ToyExample("fragrance free moisturizer", users[0].profile, "i_001", 0),
        ToyExample("retinol cream at night", users[0].profile, "i_002", 0),
        ToyExample("vitamin c serum for glow", users[0].profile, "i_003", 0),
        ToyExample("noise cancelling headset", users[1].profile, "i_010", 1),
        ToyExample("waterproof bluetooth speaker", users[1].profile, "i_011", 1),
        ToyExample("compact mechanical keyboard", users[1].profile, "i_012", 1),
        ToyExample("breathable running sneakers", users[2].profile, "i_020", 2),
        ToyExample("non slip yoga mat", users[2].profile, "i_021", 2),
        ToyExample("compression socks long flight", users[2].profile, "i_022", 2),
    ]
    return items, users, examples


def build_vocab(texts: Iterable[str], extra_tokens: list[str]) -> Vocab:
    tokens = ["<pad>", "<bos>", "<eos>", "<unk>"] + extra_tokens
    for t in texts:
        tokens.extend(basic_tokenize(t))
    return Vocab.from_tokens(tokens)


class TextSeqDataset(Dataset):
    def __init__(
        self,
        queries: list[str],
        targets: list[list[str]],
        query_vocab: Vocab,
        target_vocab: Vocab,
        add_user_profile: bool,
        user_profiles: list[str] | None = None,
    ) -> None:
        if add_user_profile and not user_profiles:
            raise ValueError("user_profiles required when add_user_profile=True")
        self.queries = queries
        self.targets = targets
        self.query_vocab = query_vocab
        self.target_vocab = target_vocab
        self.add_user_profile = add_user_profile
        self.user_profiles = user_profiles

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        q = self.queries[idx]
        if self.add_user_profile:
            q = f"user: {self.user_profiles[idx]} query: {q}"

        q_ids = [self.query_vocab.token_to_id["<bos>"]] + self.query_vocab.encode(
            basic_tokenize(q)
        ) + [self.query_vocab.token_to_id["<eos>"]]

        t_tokens = self.targets[idx]
        t_ids = [self.target_vocab.token_to_id["<bos>"]] + self.target_vocab.encode(
            t_tokens
        ) + [self.target_vocab.token_to_id["<eos>"]]

        return {
            "query_ids": torch.tensor(q_ids, dtype=torch.long),
            "target_ids": torch.tensor(t_ids, dtype=torch.long),
        }


def pad_1d(seqs: list[torch.Tensor], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(x.numel() for x in seqs)
    out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), max_len), dtype=torch.bool)
    for i, x in enumerate(seqs):
        out[i, : x.numel()] = x
        mask[i, : x.numel()] = True
    return out, mask


def collate_text_batch(batch: list[dict[str, torch.Tensor]], pad_id_q: int, pad_id_t: int):
    q, q_mask = pad_1d([b["query_ids"] for b in batch], pad_id_q)
    t, t_mask = pad_1d([b["target_ids"] for b in batch], pad_id_t)
    return {
        "query_ids": q,
        "query_mask": q_mask,
        "target_ids": t,
        "target_mask": t_mask,
    }


def build_click_reward_matrix(examples: list[ToyExample], item_ids: list[str]) -> np.ndarray:
    # rows: examples, cols: items
    id_to_col = {iid: i for i, iid in enumerate(item_ids)}
    mat = np.zeros((len(examples), len(item_ids)), dtype=np.float32)
    for r, ex in enumerate(examples):
        mat[r, id_to_col[ex.item_id]] = 1.0
    return mat
