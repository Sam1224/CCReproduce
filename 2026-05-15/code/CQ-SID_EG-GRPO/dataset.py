import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SidConfig:
    n_levels: int = 3
    # Level-1 is category-aware: each category has its own codebook of size k1.
    k1: int = 16
    k2: int = 32
    k3: int = 32


@dataclass(frozen=True)
class ToyConfig:
    seed: int = 42
    n_categories: int = 8
    n_brands: int = 20
    n_colors: int = 8
    n_materials: int = 10
    n_items: int = 500
    n_users: int = 200

    # For each item, generate multiple queries.
    queries_per_item: int = 2


class Vocab:
    def __init__(self, tokens: List[str]):
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: List[str] = []
        for t in tokens:
            self.add(t)

    def add(self, token: str) -> int:
        if token in self.token_to_id:
            return self.token_to_id[token]
        idx = len(self.id_to_token)
        self.token_to_id[token] = idx
        self.id_to_token.append(token)
        return idx

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.token_to_id[t] for t in tokens]

    def __len__(self) -> int:
        return len(self.id_to_token)


def build_vocab(toy: ToyConfig, sid: SidConfig) -> Tuple[Vocab, Dict[str, int]]:
    special = ["<pad>", "<bos>", "<eos>"]
    base = []

    cats = [f"cat_{i}" for i in range(toy.n_categories)]
    brands = [f"brand_{i}" for i in range(toy.n_brands)]
    colors = [f"color_{i}" for i in range(toy.n_colors)]
    materials = [f"mat_{i}" for i in range(toy.n_materials)]

    base += cats + brands + colors + materials
    base += ["buy", "cheap", "premium", "for", "gift", "new", "sale"]

    # SID tokens
    sid_tokens = []
    for c in range(toy.n_categories):
        for i in range(sid.k1):
            sid_tokens.append(f"SID_L1_{c}_{i}")
    for i in range(sid.k2):
        sid_tokens.append(f"SID_L2_{i}")
    for i in range(sid.k3):
        sid_tokens.append(f"SID_L3_{i}")

    vocab = Vocab(special + base + sid_tokens)

    specials = {
        "pad": vocab.token_to_id["<pad>"],
        "bos": vocab.token_to_id["<bos>"],
        "eos": vocab.token_to_id["<eos>"],
    }
    return vocab, specials


@dataclass
class Item:
    item_id: int
    category: int
    brand: int
    color: int
    material: int
    title_tokens: List[str]


@dataclass
class User:
    user_id: int
    preferred_brand: int


def _rng(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def generate_toy_catalog(toy: ToyConfig) -> Tuple[List[Item], List[User]]:
    _rng(toy.seed)

    items: List[Item] = []
    for item_id in range(toy.n_items):
        category = random.randrange(toy.n_categories)
        brand = random.randrange(toy.n_brands)
        color = random.randrange(toy.n_colors)
        material = random.randrange(toy.n_materials)

        title_tokens = [
            f"cat_{category}",
            f"brand_{brand}",
            f"color_{color}",
            f"mat_{material}",
        ]
        items.append(
            Item(
                item_id=item_id,
                category=category,
                brand=brand,
                color=color,
                material=material,
                title_tokens=title_tokens,
            )
        )

    users: List[User] = []
    for user_id in range(toy.n_users):
        preferred_brand = random.randrange(toy.n_brands)
        users.append(User(user_id=user_id, preferred_brand=preferred_brand))

    return items, users


def make_query(item: Item, user: User, with_user: bool = True) -> List[str]:
    # In the paper, category constraint is explicitly used.
    # Here we always prefix category token to enforce that constraint.
    q = [f"cat_{item.category}", "buy"]

    # Add a couple of attributes.
    if random.random() < 0.7:
        q.append(f"color_{item.color}")
    if random.random() < 0.5:
        q.append(f"mat_{item.material}")

    # Add a (possibly user-conditioned) brand preference token.
    if with_user and random.random() < 0.6:
        q.append("for")
        q.append(f"brand_{user.preferred_brand}")

    if random.random() < 0.3:
        q.append("sale")

    return q


@dataclass
class Example:
    user_id: int
    item_id: int
    src_tokens: List[str]


def build_examples(items: List[Item], users: List[User], toy: ToyConfig, with_user: bool) -> List[Example]:
    _rng(toy.seed + (1 if with_user else 0))

    examples: List[Example] = []
    for item in items:
        for _ in range(toy.queries_per_item):
            user = random.choice(users)
            q = make_query(item=item, user=user, with_user=with_user)
            examples.append(Example(user_id=user.user_id, item_id=item.item_id, src_tokens=q))

    random.shuffle(examples)
    return examples


class SftDataset(Dataset):
    def __init__(
        self,
        examples: List[Example],
        items: List[Item],
        sid_tokens_by_item: Dict[int, List[str]],
        vocab: Vocab,
        specials: Dict[str, int],
        max_src_len: int = 32,
    ):
        self.examples = examples
        self.items = {it.item_id: it for it in items}
        self.sid_tokens_by_item = sid_tokens_by_item
        self.vocab = vocab
        self.specials = specials
        self.max_src_len = max_src_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        src = ex.src_tokens[: self.max_src_len]
        src_ids = self.vocab.encode(src)

        sid_tokens = self.sid_tokens_by_item[ex.item_id]
        # Target: <bos> SID_L1_x SID_L2_y SID_L3_z <eos>
        tgt = ["<bos>"] + sid_tokens + ["<eos>"]
        tgt_ids = self.vocab.encode(tgt)

        return {
            "user_id": ex.user_id,
            "item_id": ex.item_id,
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "tgt_ids": torch.tensor(tgt_ids, dtype=torch.long),
        }


def collate_sft(batch: List[dict], pad_id: int):
    src_lens = [b["src_ids"].numel() for b in batch]
    tgt_lens = [b["tgt_ids"].numel() for b in batch]

    max_src = max(src_lens)
    max_tgt = max(tgt_lens)

    src = torch.full((len(batch), max_src), pad_id, dtype=torch.long)
    tgt = torch.full((len(batch), max_tgt), pad_id, dtype=torch.long)

    for i, b in enumerate(batch):
        src[i, : b["src_ids"].numel()] = b["src_ids"]
        tgt[i, : b["tgt_ids"].numel()] = b["tgt_ids"]

    return {
        "src": src,
        "tgt": tgt,
        "src_lens": torch.tensor(src_lens, dtype=torch.long),
        "tgt_lens": torch.tensor(tgt_lens, dtype=torch.long),
        "user_id": torch.tensor([b["user_id"] for b in batch], dtype=torch.long),
        "item_id": torch.tensor([b["item_id"] for b in batch], dtype=torch.long),
    }


def split_train_test(examples: List[Example], train_ratio: float = 0.9):
    n_train = int(math.floor(len(examples) * train_ratio))
    return examples[:n_train], examples[n_train:]
