from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset


CATEGORIES = {
    "sneakers": {
        "aliases": ["sneakers", "running shoes", "sport shoes"],
        "brands": ["NovaRun", "FlyStep", "UrbanSprint"],
        "attrs": {
            "audience": ["men", "women", "kids"],
            "style": ["breathable", "lightweight", "cushioned"],
            "color": ["black", "white", "red"],
        },
    },
    "lipstick": {
        "aliases": ["lipstick", "lip color", "matte lipstick"],
        "brands": ["VelvetHue", "RoseMuse", "GlowLab"],
        "attrs": {
            "finish": ["matte", "glossy", "velvet"],
            "tone": ["warm red", "pink nude", "berry"],
            "color": ["red", "pink", "brown"],
        },
    },
    "phone": {
        "aliases": ["phone", "smartphone", "mobile phone"],
        "brands": ["Orion", "Pulse", "Nebula"],
        "attrs": {
            "memory": ["128gb", "256gb", "512gb"],
            "camera": ["portrait", "night shot", "vlog"],
            "color": ["black", "silver", "blue"],
        },
    },
    "shampoo": {
        "aliases": ["shampoo", "hair shampoo", "repair shampoo"],
        "brands": ["PureDrop", "SilkRoot", "FreshMoss"],
        "attrs": {
            "hair": ["oily hair", "dry hair", "damaged hair"],
            "effect": ["repair", "anti dandruff", "refreshing"],
            "size": ["200ml", "400ml", "800ml"],
        },
    },
    "desk": {
        "aliases": ["desk", "office desk", "computer desk"],
        "brands": ["OakPlus", "FlexiWork", "MetroDesk"],
        "attrs": {
            "material": ["wood", "metal", "engineered wood"],
            "feature": ["standing", "compact", "drawer"],
            "color": ["white", "brown", "black"],
        },
    },
}

TAIL_TEMPLATES = [
    "{alias} for {primary_attr}",
    "{brand} {alias} {secondary_attr}",
    "best {alias} with {primary_attr} {secondary_attr}",
]

HEAD_TEMPLATES = [
    "{alias}",
    "{brand} {alias}",
    "buy {alias}",
]


@dataclass(frozen=True)
class Item:
    item_id: str
    category: str
    brand: str
    attrs: dict[str, str]
    popularity: float
    channel_bias: dict[str, float]

    @property
    def text(self) -> str:
        attr_text = " ".join(self.attrs.values())
        return f"{self.brand} {self.category} {attr_text}".strip()


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str
    category: str
    focus_attrs: tuple[str, ...]
    brand: str | None
    head_tail: str


@dataclass(frozen=True)
class TripletRecord:
    query_text: str
    pos_text: str
    neg_text: str
    difficulty: int
    pos_grade: int
    pos_confidence: float
    neg_confidence: float
    head_tail: str


@dataclass(frozen=True)
class EvalCandidate:
    item_text: str
    grade: int
    difficulty: int
    channels: tuple[str, ...]


@dataclass(frozen=True)
class EvalRecord:
    query_text: str
    head_tail: str
    candidates: tuple[EvalCandidate, ...]


def stable_hash(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest(), 16)


def text_to_ids(text: str, *, max_len: int = 20, vocab_size: int = 4096) -> list[int]:
    tokens = text.lower().replace("/", " ").replace("-", " ").split()
    ids: list[int] = []
    for token in tokens[:max_len]:
        ids.append(1 + stable_hash(token) % (vocab_size - 1))
    if not ids:
        ids = [1]
    if len(ids) < max_len:
        ids.extend([0] * (max_len - len(ids)))
    return ids


def overlap_ratio(query_text: str, item_text: str) -> float:
    query_tokens = set(query_text.lower().split())
    item_tokens = set(item_text.lower().split())
    if not query_tokens:
        return 0.0
    return len(query_tokens & item_tokens) / len(query_tokens)


def oracle_score(query: Query, item: Item) -> float:
    score = 0.05
    if item.category == query.category:
        score += 0.4
    if query.brand and item.brand == query.brand:
        score += 0.15
    values = set(item.attrs.values())
    score += 0.18 * sum(attr in values for attr in query.focus_attrs)
    lexical = overlap_ratio(query.text, item.text)
    score += 0.22 * lexical
    return min(score, 1.0)


def lexical_score(query: Query, item: Item, rng: random.Random) -> float:
    return 0.7 * overlap_ratio(query.text, item.text) + 0.15 * (item.category == query.category) + item.channel_bias["lexical"] + rng.uniform(-0.04, 0.04)


def taxonomy_score(query: Query, item: Item, rng: random.Random) -> float:
    match_attrs = sum(attr in set(item.attrs.values()) for attr in query.focus_attrs)
    return 0.5 * (item.category == query.category) + 0.18 * match_attrs + 0.12 * item.popularity + item.channel_bias["taxonomy"] + rng.uniform(-0.05, 0.05)


def behavior_score(query: Query, item: Item, rng: random.Random) -> float:
    shared = sum(attr in set(item.attrs.values()) for attr in query.focus_attrs)
    score = 0.3 * (item.category == query.category) + 0.22 * item.popularity + 0.1 * shared + item.channel_bias["behavior"]
    if query.head_tail == "tail":
        score -= 0.08
    return score + rng.uniform(-0.06, 0.06)


def grade_from_score(score: float) -> int:
    if score >= 0.82:
        return 4
    if score >= 0.62:
        return 3
    if score >= 0.42:
        return 2
    if score >= 0.22:
        return 1
    return 0


def assign_difficulty(*, grade: int, confidence: float, channels: tuple[str, ...], best_rank: int) -> int:
    if grade == 4 and len(channels) == 3 and best_rank <= 3 and confidence >= 0.82:
        return 1
    if grade >= 3 and len(channels) >= 2 and best_rank <= 8:
        return 2
    if grade >= 3 and len(channels) == 1:
        return 3
    if grade <= 1 and best_rank <= 5:
        return 4
    return 5


def llm_cascade(query: Query, item: Item, oracle: float) -> tuple[int, float, str]:
    lexical = overlap_ratio(query.text, item.text)
    judge1 = min(1.0, 0.75 * lexical + 0.08 * (item.category == query.category))
    conf1 = 0.55 + abs(judge1 - 0.5)
    if conf1 >= 0.86:
        return grade_from_score(judge1), conf1, "judge1"

    attr_match = sum(attr in set(item.attrs.values()) for attr in query.focus_attrs)
    judge2 = min(1.0, 0.3 * lexical + 0.35 * (item.category == query.category) + 0.18 * attr_match + 0.08 * (query.brand is not None and item.brand == query.brand))
    conf2 = 0.58 + 0.08 * attr_match + abs(judge2 - 0.5)
    if conf2 >= 0.82:
        return grade_from_score(judge2), min(conf2, 0.97), "judge2"

    judge3 = min(1.0, 0.15 * judge2 + 0.85 * oracle)
    conf3 = 0.86 + 0.1 * abs(judge3 - 0.5)
    return grade_from_score(judge3), min(conf3, 0.99), "judge3"


def build_items(rng: random.Random, per_category: int = 72) -> list[Item]:
    items: list[Item] = []
    for category, meta in CATEGORIES.items():
        attr_keys = list(meta["attrs"].keys())
        for index in range(per_category):
            brand = meta["brands"][index % len(meta["brands"])]
            attrs = {key: meta["attrs"][key][(index + offset) % len(meta["attrs"][key])] for offset, key in enumerate(attr_keys)}
            items.append(
                Item(
                    item_id=f"{category}-{index}",
                    category=category,
                    brand=brand,
                    attrs=attrs,
                    popularity=0.2 + 0.8 * rng.random(),
                    channel_bias={
                        "lexical": rng.uniform(-0.06, 0.08),
                        "taxonomy": rng.uniform(-0.06, 0.08),
                        "behavior": rng.uniform(-0.08, 0.1),
                    },
                )
            )
    return items


def build_queries(rng: random.Random, per_category: int = 44) -> list[Query]:
    queries: list[Query] = []
    for category, meta in CATEGORIES.items():
        attr_keys = list(meta["attrs"].keys())
        for index in range(per_category):
            alias = meta["aliases"][index % len(meta["aliases"])]
            brand = meta["brands"][index % len(meta["brands"])] if index % 3 == 0 else None
            primary = meta["attrs"][attr_keys[0]][index % len(meta["attrs"][attr_keys[0]])]
            secondary = meta["attrs"][attr_keys[1]][(index + 1) % len(meta["attrs"][attr_keys[1]])]
            head_tail = "tail" if index % 4 else "head"
            template = rng.choice(TAIL_TEMPLATES if head_tail == "tail" else HEAD_TEMPLATES)
            text = template.format(alias=alias, primary_attr=primary, secondary_attr=secondary, brand=brand or "")
            queries.append(
                Query(
                    query_id=f"{category}-q{index}",
                    text=" ".join(text.split()),
                    category=category,
                    focus_attrs=(primary, secondary),
                    brand=brand,
                    head_tail=head_tail,
                )
            )
    rng.shuffle(queries)
    return queries


def topk(scores: Iterable[tuple[str, float]], k: int) -> list[str]:
    return [item_id for item_id, _ in sorted(scores, key=lambda pair: pair[1], reverse=True)[:k]]


def build_query_records(query: Query, items: list[Item], rng: random.Random, topk_per_channel: int) -> tuple[list[dict], dict]:
    by_id = {item.item_id: item for item in items}
    lexical_ids = topk(((item.item_id, lexical_score(query, item, rng)) for item in items), topk_per_channel)
    taxonomy_ids = topk(((item.item_id, taxonomy_score(query, item, rng)) for item in items), topk_per_channel)
    behavior_ids = topk(((item.item_id, behavior_score(query, item, rng)) for item in items), topk_per_channel)

    union_ids = list(dict.fromkeys(lexical_ids + taxonomy_ids + behavior_ids))
    records: list[dict] = []
    positives: list[dict] = []
    negatives: list[dict] = []

    for item_id in union_ids:
        item = by_id[item_id]
        oracle = oracle_score(query, item)
        grade, confidence, judge_used = llm_cascade(query, item, oracle)
        ranks = {
            "lexical": lexical_ids.index(item_id) + 1 if item_id in lexical_ids else None,
            "taxonomy": taxonomy_ids.index(item_id) + 1 if item_id in taxonomy_ids else None,
            "behavior": behavior_ids.index(item_id) + 1 if item_id in behavior_ids else None,
        }
        channels = tuple(name for name, rank in ranks.items() if rank is not None)
        best_rank = min(rank for rank in ranks.values() if rank is not None)
        difficulty = assign_difficulty(grade=grade, confidence=confidence, channels=channels, best_rank=best_rank)
        record = {
            "query_id": query.query_id,
            "query_text": query.text,
            "head_tail": query.head_tail,
            "item_id": item.item_id,
            "item_text": item.text,
            "grade": grade,
            "confidence": round(confidence, 4),
            "judge_used": judge_used,
            "difficulty": difficulty,
            "channels": channels,
            "best_rank": best_rank,
        }
        records.append(record)
        if grade >= 3:
            positives.append(record)
        if grade <= 1:
            negatives.append(record)

    positives.sort(key=lambda row: (-row["grade"], row["difficulty"], -row["confidence"]))
    negatives.sort(key=lambda row: (row["difficulty"], row["best_rank"], row["confidence"]))
    return records, {"positives": positives, "negatives": negatives}


def split_queries(queries: list[Query]) -> dict[str, list[Query]]:
    total = len(queries)
    train_cut = int(total * 0.72)
    dev_cut = int(total * 0.86)
    return {
        "train": queries[:train_cut],
        "dev": queries[train_cut:dev_cut],
        "test": queries[dev_cut:],
    }


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def ensure_toy_data(data_dir: str | Path, *, seed: int = 42, topk_per_channel: int = 20, force: bool = False) -> Path:
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    required = [root / "items.jsonl", root / "train_triples.jsonl", root / "dev.jsonl", root / "test.jsonl"]
    if not force and all(path.exists() for path in required):
        return root

    rng = random.Random(seed)
    items = build_items(rng)
    queries = build_queries(rng)
    query_splits = split_queries(queries)

    triplets: list[dict] = []
    items_rows = [{"item_id": item.item_id, "item_text": item.text} for item in items]
    eval_rows_by_split: dict[str, list[dict]] = {"dev": [], "test": []}

    for split, query_list in query_splits.items():
        for query in query_list:
            records, buckets = build_query_records(query, items, rng, topk_per_channel=topk_per_channel)
            eval_row = {
                "query_text": query.text,
                "head_tail": query.head_tail,
                "candidates": [
                    {
                        "item_text": row["item_text"],
                        "grade": row["grade"],
                        "difficulty": row["difficulty"],
                        "channels": row["channels"],
                    }
                    for row in records
                ],
            }
            if split == "train":
                positives = buckets["positives"][:4]
                hard_negs = sorted(buckets["negatives"], key=lambda row: (row["difficulty"], row["best_rank"]))[:6]
                if not positives or not hard_negs:
                    continue
                for pos in positives:
                    neg = hard_negs[(stable_hash(pos["item_id"] + query.query_id) % len(hard_negs))]
                    triplets.append(
                        {
                            "query_text": query.text,
                            "pos_text": pos["item_text"],
                            "neg_text": neg["item_text"],
                            "difficulty": min(5, int(round((pos["difficulty"] + neg["difficulty"]) / 2))),
                            "pos_grade": pos["grade"],
                            "pos_confidence": pos["confidence"],
                            "neg_confidence": neg["confidence"],
                            "head_tail": query.head_tail,
                        }
                    )
            else:
                eval_rows_by_split[split].append(eval_row)

    write_jsonl(root / "items.jsonl", items_rows)
    write_jsonl(root / "train_triples.jsonl", triplets)
    write_jsonl(root / "dev.jsonl", eval_rows_by_split["dev"])
    write_jsonl(root / "test.jsonl", eval_rows_by_split["test"])
    return root


class RetrievalTripletDataset(Dataset):
    def __init__(self, rows: list[dict], *, max_difficulty: int | None = None, max_len: int = 20, vocab_size: int = 4096):
        if max_difficulty is not None:
            rows = [row for row in rows if row["difficulty"] <= max_difficulty]
        self.rows = rows
        self.max_len = max_len
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        return {
            "q_ids": torch.tensor(text_to_ids(row["query_text"], max_len=self.max_len, vocab_size=self.vocab_size), dtype=torch.long),
            "pos_ids": torch.tensor(text_to_ids(row["pos_text"], max_len=self.max_len, vocab_size=self.vocab_size), dtype=torch.long),
            "neg_ids": torch.tensor(text_to_ids(row["neg_text"], max_len=self.max_len, vocab_size=self.vocab_size), dtype=torch.long),
            "difficulty": torch.tensor(row["difficulty"], dtype=torch.long),
            "pos_grade": torch.tensor(row["pos_grade"], dtype=torch.float32),
            "head_tail": torch.tensor(1 if row["head_tail"] == "tail" else 0, dtype=torch.long),
        }


def load_train_rows(data_dir: str | Path) -> list[dict]:
    return read_jsonl(Path(data_dir) / "train_triples.jsonl")


def load_eval_records(data_dir: str | Path, split: str) -> list[EvalRecord]:
    rows = read_jsonl(Path(data_dir) / f"{split}.jsonl")
    records: list[EvalRecord] = []
    for row in rows:
        candidates = tuple(
            EvalCandidate(
                item_text=candidate["item_text"],
                grade=int(candidate["grade"]),
                difficulty=int(candidate["difficulty"]),
                channels=tuple(candidate["channels"]),
            )
            for candidate in row["candidates"]
        )
        records.append(EvalRecord(query_text=row["query_text"], head_tail=row["head_tail"], candidates=candidates))
    return records
