from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class PairExample:
    query: str
    item: str
    item_id: int


@dataclass
class RankPair:
    query: str
    item: str
    item_id: int
    label: int
    pos_item_id: int


@dataclass
class GroupEval:
    query: str
    candidate_items: List[str]
    candidate_ids: List[int]
    labels: List[int]
    pos_item_id: int


@dataclass
class DatasetBundle:
    stage1_train: List[PairExample]
    stage1_val: List[PairExample]
    rank_train: List[RankPair]
    rank_val: List[RankPair]
    eval_groups: List[GroupEval]


_COLORS = [
    "black",
    "white",
    "red",
    "blue",
    "green",
    "yellow",
    "beige",
    "pink",
]
_MATERIALS = ["cotton", "leather", "denim", "wool", "silk", "linen"]
_SIZES = ["xs", "s", "m", "l", "xl"]
_CATS = ["tshirt", "jacket", "dress", "shoes", "bag", "hoodie"]
_STYLES = ["casual", "formal", "sport", "vintage"]


def _item_text(color: str, material: str, cat: str, size: str, style: str) -> str:
    # Intentionally add high lexical overlap.
    return f"{style} {color} {material} {cat} size {size} premium {cat}"


def _query_text(color: str, cat: str, size: str | None, style: str | None) -> str:
    toks = ["buy", "a", color, cat]
    if style is not None:
        toks.insert(2, style)
    if size is not None:
        toks += ["size", size]
    return " ".join(toks)


def build_toy_dataset(
    seed: int = 7,
    num_items: int = 1200,
    num_queries: int = 2500,
    group_k: int = 10,
) -> DatasetBundle:
    """Synthetic e-commerce relevance dataset.

    Designed to mimic:
    - short ambiguous queries,
    - high lexical overlap among product titles,
    - fine-grained attribute distinctions.
    """

    rng = random.Random(seed)

    # ----- Build catalog -----
    items: List[Tuple[str, str, str, str, str]] = []
    index: Dict[Tuple[str, str, str, str, str], int] = {}
    for item_id in range(num_items):
        color = rng.choice(_COLORS)
        material = rng.choice(_MATERIALS)
        cat = rng.choice(_CATS)
        size = rng.choice(_SIZES)
        style = rng.choice(_STYLES)
        key = (color, material, cat, size, style)
        if key in index:
            continue
        index[key] = len(items)
        items.append(key)

    def pick_item(color: str, cat: str, size: str | None, style: str | None) -> int:
        candidates = []
        for (c, m, k, s, st), idx in index.items():
            if c != color or k != cat:
                continue
            if size is not None and s != size:
                continue
            if style is not None and st != style:
                continue
            candidates.append(idx)
        if candidates:
            return rng.choice(candidates)
        return rng.randrange(len(items))

    # ----- Build queries & pairs -----
    stage1_pairs: List[PairExample] = []
    rank_pairs: List[RankPair] = []
    eval_groups: List[GroupEval] = []

    for _ in range(num_queries):
        color = rng.choice(_COLORS)
        cat = rng.choice(_CATS)
        size = rng.choice(_SIZES)
        style = rng.choice(_STYLES)

        # Drop attributes to simulate ambiguity.
        size_q = size if rng.random() > 0.35 else None
        style_q = style if rng.random() > 0.35 else None

        q = _query_text(color=color, cat=cat, size=size_q, style=style_q)
        pos_id = pick_item(color=color, cat=cat, size=size, style=style)
        pos_key = items[pos_id]
        pos_text = _item_text(*pos_key)

        stage1_pairs.append(PairExample(query=q, item=pos_text, item_id=pos_id))

        # Create ranking pairs: 1 positive + negatives
        rank_pairs.append(RankPair(query=q, item=pos_text, item_id=pos_id, label=1, pos_item_id=pos_id))

        # Hard negatives: match (color, cat) but mismatch size or style.
        hard_ids: List[int] = []
        for (c, _m, k, s, st), idx in index.items():
            if c == color and k == cat and (s != size or st != style):
                hard_ids.append(idx)
        rng.shuffle(hard_ids)
        hard_ids = hard_ids[: max(3, group_k // 3)]

        # Easy negatives
        easy_ids = [rng.randrange(len(items)) for _ in range(group_k)]

        neg_ids = (hard_ids + easy_ids)[: group_k - 1]

        candidate_ids = [pos_id] + neg_ids
        candidate_items = [pos_text]
        labels = [1]
        for nid in neg_ids:
            candidate_items.append(_item_text(*items[nid]))
            labels.append(0)
            rank_pairs.append(RankPair(query=q, item=candidate_items[-1], item_id=nid, label=0, pos_item_id=pos_id))

        eval_groups.append(GroupEval(query=q, candidate_items=candidate_items, candidate_ids=candidate_ids, labels=labels, pos_item_id=pos_id))

    # ----- Split -----
    rng.shuffle(stage1_pairs)
    rng.shuffle(rank_pairs)
    rng.shuffle(eval_groups)

    s1_cut = int(0.9 * len(stage1_pairs))
    r_cut = int(0.9 * len(rank_pairs))

    return DatasetBundle(
        stage1_train=stage1_pairs[:s1_cut],
        stage1_val=stage1_pairs[s1_cut:],
        rank_train=rank_pairs[:r_cut],
        rank_val=rank_pairs[r_cut:],
        eval_groups=eval_groups[: min(600, len(eval_groups))],
    )
