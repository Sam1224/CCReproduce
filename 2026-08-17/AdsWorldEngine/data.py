from __future__ import annotations

from dataclasses import dataclass
from random import Random

import torch
from torch.utils.data import Dataset


TOPICS = ["electronics", "fashion", "fitness", "beauty", "home"]
BRANDS = ["nova", "lumi", "pulse", "craft", "zena", "orbit"]
DIALOG_TOKENS = {
    token: index
    for index, token in enumerate(
        [
            "buy",
            "deal",
            "coupon",
            "recommend",
            "gift",
            "electronics",
            "fashion",
            "fitness",
            "beauty",
            "home",
            "compare",
            "price",
            "routine",
            "setup",
            "guide",
            "urgent",
            "casual",
            "brand_safe",
        ]
    )
}


@dataclass
class AdItem:
    ad_id: str
    topic: str
    brand: str
    bid: float
    quality: float


@dataclass
class ConversationExample:
    query_tokens: list[str]
    reply_tokens: list[str]
    history_tokens: list[str]
    target_topic: str
    trigger_label: int


@dataclass
class SlateExample:
    conversation: ConversationExample
    candidate_ads: list[AdItem]


def dialog_to_bow(example: ConversationExample) -> torch.Tensor:
    vec = torch.zeros(len(DIALOG_TOKENS), dtype=torch.float32)
    for token in example.query_tokens + example.reply_tokens + example.history_tokens:
        if token in DIALOG_TOKENS:
            vec[DIALOG_TOKENS[token]] += 1.0
    return vec


def ad_to_vector(ad: AdItem) -> torch.Tensor:
    topic_vec = [1.0 if ad.topic == topic else 0.0 for topic in TOPICS]
    brand_vec = [1.0 if ad.brand == brand else 0.0 for brand in BRANDS]
    numeric = [ad.bid / 5.0, ad.quality]
    return torch.tensor(topic_vec + brand_vec + numeric, dtype=torch.float32)


def build_ads(seed: int = 23) -> list[AdItem]:
    rng = Random(seed)
    ads: list[AdItem] = []
    for topic in TOPICS:
        for brand in BRANDS:
            for index in range(4):
                ads.append(
                    AdItem(
                        ad_id=f"{topic[:2]}-{brand[:2]}-{index}",
                        topic=topic,
                        brand=brand,
                        bid=round(rng.uniform(0.8, 4.5), 3),
                        quality=round(rng.uniform(0.45, 0.98), 4),
                    )
                )
    return ads


def build_conversations(seed: int = 31, size: int = 240) -> list[ConversationExample]:
    rng = Random(seed)
    examples: list[ConversationExample] = []
    utility_tokens = {
        "electronics": ["buy", "compare", "electronics", "price"],
        "fashion": ["buy", "fashion", "gift", "deal"],
        "fitness": ["fitness", "recommend", "routine", "deal"],
        "beauty": ["beauty", "routine", "gift", "deal"],
        "home": ["home", "setup", "recommend", "price"],
    }
    for _ in range(size):
        topic = rng.choice(TOPICS)
        base_tokens = utility_tokens[topic][:]
        rng.shuffle(base_tokens)
        query_tokens = base_tokens[:3]
        reply_tokens = [topic, "recommend"] if rng.random() < 0.75 else ["guide", topic]
        history_tokens = ["casual"]
        trigger_label = 1
        if rng.random() < 0.25:
            query_tokens = ["guide", topic]
            reply_tokens = ["brand_safe", "casual"]
            trigger_label = 0
        if rng.random() < 0.15:
            history_tokens.append("urgent")
        examples.append(
            ConversationExample(
                query_tokens=query_tokens,
                reply_tokens=reply_tokens,
                history_tokens=history_tokens,
                target_topic=topic,
                trigger_label=trigger_label,
            )
        )
    return examples


def tool_features(example: ConversationExample, ad: AdItem) -> torch.Tensor:
    bow = set(example.query_tokens + example.reply_tokens)
    topical_overlap = 1.0 if ad.topic in bow else 0.0
    commerce_intent = 1.0 if any(token in bow for token in ["buy", "deal", "coupon", "price", "recommend"]) else 0.0
    reply_match = 1.0 if ad.topic in example.reply_tokens else 0.0
    brand_safe = 1.0 if "brand_safe" in example.reply_tokens else 0.0
    return torch.tensor([topical_overlap, commerce_intent, reply_match, brand_safe], dtype=torch.float32)


def relevance_label(example: ConversationExample, ad: AdItem) -> float:
    if not example.trigger_label:
        return 0.0
    topic_match = 1.0 if ad.topic == example.target_topic else 0.0
    bonus = 0.2 if ad.quality > 0.75 else 0.0
    return min(1.0, topic_match + bonus)


def slate_reward(example: ConversationExample, slate: list[AdItem]) -> float:
    if not slate:
        return 0.0
    relevance = sum(relevance_label(example, ad) for ad in slate) / len(slate)
    brand_diversity = len({ad.brand for ad in slate}) / len(slate)
    quality = sum(ad.quality for ad in slate) / len(slate)
    return 0.55 * relevance + 0.2 * brand_diversity + 0.25 * quality


class GateDataset(Dataset):
    def __init__(self, examples: list[ConversationExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        example = self.examples[index]
        return dialog_to_bow(example), torch.tensor(float(example.trigger_label), dtype=torch.float32)


class JudgeDataset(Dataset):
    def __init__(self, examples: list[ConversationExample], ads: list[AdItem]) -> None:
        self.rows = []
        for example in examples:
            for ad in ads:
                dialog_features = dialog_to_bow(example)
                ad_features = ad_to_vector(ad)
                extra_features = tool_features(example, ad)
                label = relevance_label(example, ad)
                self.rows.append((dialog_features, ad_features, extra_features, label))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        return self.rows[index]


class OrchestratorDataset(Dataset):
    def __init__(self, examples: list[ConversationExample], ads: list[AdItem]) -> None:
        self.slates = [SlateExample(example, ads) for example in examples]

    def __len__(self) -> int:
        return len(self.slates)

    def __getitem__(self, index: int):
        return self.slates[index]


def collate_gate(batch):
    features = torch.stack([row[0] for row in batch])
    labels = torch.tensor([row[1] for row in batch], dtype=torch.float32)
    return features, labels


def collate_judge(batch):
    dialog_features = torch.stack([row[0] for row in batch])
    ad_features = torch.stack([row[1] for row in batch])
    tool_feature_rows = torch.stack([row[2] for row in batch])
    labels = torch.tensor([row[3] for row in batch], dtype=torch.float32)
    return dialog_features, ad_features, tool_feature_rows, labels


def benchmark_conversations() -> list[ConversationExample]:
    return [
        ConversationExample(["buy", "electronics", "price"], ["electronics", "recommend"], ["urgent"], "electronics", 1),
        ConversationExample(["guide", "beauty"], ["brand_safe", "casual"], ["casual"], "beauty", 0),
        ConversationExample(["deal", "fitness", "recommend"], ["fitness", "recommend"], ["casual"], "fitness", 1),
    ]
