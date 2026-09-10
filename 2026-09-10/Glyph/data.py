import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset


ONTOLOGY = [
    "NON_SENSITIVE",
    "USER_ID",
    "EMAIL",
    "PHONE",
    "ADDRESS",
    "PAYMENT",
    "DEVICE_ID",
    "CONTENT_POLICY",
    "CREATOR_RISK",
    "PRODUCT_ATTRIBUTE",
]

TAG_TO_ID = {tag: index for index, tag in enumerate(ONTOLOGY)}

PATTERNS = {
    "EMAIL": ["email", "mail", "e_mail"],
    "PHONE": ["phone", "mobile", "tel"],
    "ADDRESS": ["address", "addr", "location", "geo"],
    "PAYMENT": ["card", "payment", "pay", "billing"],
    "DEVICE_ID": ["device", "imei", "idfa", "oaid"],
    "USER_ID": ["user", "uid", "buyer", "seller", "author"],
    "CONTENT_POLICY": ["moderation", "policy", "violation", "unsafe", "caption"],
    "CREATOR_RISK": ["creator", "influencer", "达人", "risk", "penalty"],
    "PRODUCT_ATTRIBUTE": ["sku", "product", "item", "brand", "category"],
}

DESCRIPTIONS = {
    "USER_ID": ["unique user identifier", "buyer account key", "seller or author id"],
    "EMAIL": ["email address for account communication", "contact email"],
    "PHONE": ["phone number used for verification", "mobile contact"],
    "ADDRESS": ["delivery address and geo location", "shipping location"],
    "PAYMENT": ["payment token or billing attribute", "masked card attribute"],
    "DEVICE_ID": ["device advertising identifier", "mobile device id"],
    "CONTENT_POLICY": ["content moderation label from policy review", "unsafe caption flag"],
    "CREATOR_RISK": ["creator governance risk level", "达人 penalty history"],
    "PRODUCT_ATTRIBUTE": ["product catalogue attribute", "sku category and brand"],
    "NON_SENSITIVE": ["non sensitive operational metric", "aggregated traffic counter"],
}

CATALOGS = ["ads", "commerce", "creator", "trust", "growth"]
DBS = ["profile", "orders", "content", "risk", "events"]
TABLES = ["users", "videos", "shops", "reviews", "sessions", "items"]


@dataclass(frozen=True)
class ColumnExample:
    metadata_key: str
    description: str
    line_of_business: str
    tags: Tuple[str, ...]


def _make_column(tag: str, index: int) -> ColumnExample:
    rng = random.Random(index * 997 + len(tag))
    keyword = rng.choice(PATTERNS.get(tag, ["metric"]))
    catalog = rng.choice(CATALOGS)
    database = rng.choice(DBS)
    table = rng.choice(TABLES)
    column = f"{keyword}_{rng.choice(['id', 'value', 'score', 'flag', 'hash'])}_{index % 37}"
    description = rng.choice(DESCRIPTIONS[tag])
    tags = (tag,) if tag != "NON_SENSITIVE" else ("NON_SENSITIVE",)
    if tag in {"USER_ID", "DEVICE_ID", "PAYMENT"} and rng.random() < 0.35:
        tags = tuple(sorted(set(tags + ("CREATOR_RISK",))))
    return ColumnExample(
        metadata_key=f"{catalog}.{database}.{table}.{column}",
        description=description,
        line_of_business=catalog,
        tags=tags,
    )


def build_examples(samples_per_tag: int = 80, seed: int = 7) -> List[ColumnExample]:
    examples: List[ColumnExample] = []
    for tag in ONTOLOGY:
        for offset in range(samples_per_tag):
            examples.append(_make_column(tag, offset + seed * 1000))
    random.Random(seed).shuffle(examples)
    return examples


class CharTokenizer:
    def __init__(self, max_length: int = 96):
        alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-:/ 中文达人内容治理违规风险支付地址邮箱手机商品类目品牌"
        self.stoi: Dict[str, int] = {char: idx + 2 for idx, char in enumerate(dict.fromkeys(alphabet))}
        self.pad_id = 0
        self.unk_id = 1
        self.max_length = max_length

    def encode(self, text: str) -> torch.Tensor:
        ids = [self.stoi.get(char, self.unk_id) for char in text[: self.max_length]]
        ids += [self.pad_id] * (self.max_length - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi) + 2


class GlyphDataset(Dataset):
    def __init__(self, examples: List[ColumnExample], tokenizer: CharTokenizer):
        self.examples = examples
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        example = self.examples[index]
        text = f"{example.metadata_key} [SEP] {example.description} [LOB] {example.line_of_business}"
        labels = torch.zeros(len(ONTOLOGY), dtype=torch.float32)
        for tag in example.tags:
            labels[TAG_TO_ID[tag]] = 1.0
        primary = TAG_TO_ID[example.tags[0]]
        return {
            "input_ids": self.tokenizer.encode(text),
            "labels": labels,
            "primary_tag": torch.tensor(primary, dtype=torch.long),
        }


def split_dataset(samples_per_tag: int = 80, seed: int = 7):
    tokenizer = CharTokenizer()
    examples = build_examples(samples_per_tag=samples_per_tag, seed=seed)
    cut = int(len(examples) * 0.8)
    return GlyphDataset(examples[:cut], tokenizer), GlyphDataset(examples[cut:], tokenizer), tokenizer


def collate_fn(batch):
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
        "primary_tag": torch.stack([item["primary_tag"] for item in batch]),
    }
