from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ProductPair:
    record_id: str
    query_title: str
    query_caption: str
    candidate_title: str
    candidate_caption: str
    query_image: Sequence[float]
    candidate_image: Sequence[float]
    label: int
    ambiguity: float

    @property
    def query_text(self) -> str:
        return f"{self.query_title} {self.query_caption}"

    @property
    def candidate_text(self) -> str:
        return f"{self.candidate_title} {self.candidate_caption}"


class Vocabulary:
    def __init__(self, texts: Iterable[str], min_freq: int = 1) -> None:
        counts: Dict[str, int] = {}
        for text in texts:
            for token in self.tokenize(text):
                counts[token] = counts.get(token, 0) + 1
        self.token_to_id = {"<pad>": 0, "<unk>": 1}
        for token, count in sorted(counts.items()):
            if count >= min_freq:
                self.token_to_id[token] = len(self.token_to_id)

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text.lower())

    def encode(self, text: str, max_length: int) -> torch.Tensor:
        token_ids = [self.token_to_id.get(token, 1) for token in self.tokenize(text)[:max_length]]
        if len(token_ids) < max_length:
            token_ids.extend([0] * (max_length - len(token_ids)))
        return torch.tensor(token_ids, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.token_to_id)


class ProductLinkingDataset(Dataset):
    def __init__(self, records: Sequence[ProductPair], vocab: Vocabulary, max_length: int = 32) -> None:
        self.records = list(records)
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        record = self.records[index]
        return {
            "record_id": record.record_id,
            "query_ids": self.vocab.encode(record.query_text, self.max_length),
            "candidate_ids": self.vocab.encode(record.candidate_text, self.max_length),
            "query_image": torch.tensor(record.query_image, dtype=torch.float32),
            "candidate_image": torch.tensor(record.candidate_image, dtype=torch.float32),
            "label": torch.tensor(float(record.label), dtype=torch.float32),
            "ambiguity": torch.tensor(record.ambiguity, dtype=torch.float32),
        }


def collate_product_pairs(batch: Sequence[Dict[str, torch.Tensor | str]]) -> Dict[str, torch.Tensor | List[str]]:
    return {
        "record_id": [str(item["record_id"]) for item in batch],
        "query_ids": torch.stack([item["query_ids"] for item in batch]),
        "candidate_ids": torch.stack([item["candidate_ids"] for item in batch]),
        "query_image": torch.stack([item["query_image"] for item in batch]),
        "candidate_image": torch.stack([item["candidate_image"] for item in batch]),
        "label": torch.stack([item["label"] for item in batch]),
        "ambiguity": torch.stack([item["ambiguity"] for item in batch]),
    }


def build_toy_records(seed: int = 7) -> List[ProductPair]:
    random.seed(seed)
    base_products = [
        ("red running shoes", "mesh sneaker breathable sports", "红色 跑鞋 透气 运动鞋"),
        ("ceramic coffee mug", "large handle office cup", "陶瓷 咖啡杯 办公 水杯"),
        ("wireless ear buds", "noise cancelling bluetooth", "无线 耳机 降噪 蓝牙"),
        ("cotton summer dress", "floral lightweight women", "棉质 夏裙 碎花 女装"),
        ("stainless water bottle", "insulated outdoor thermos", "不锈钢 保温杯 户外"),
        ("phone camera tripod", "portable creator stand", "手机 三脚架 达人 拍摄"),
        ("led beauty mirror", "makeup light live commerce", "补光 化妆镜 直播 带货"),
        ("pet grooming brush", "cat dog cleaning comb", "宠物 梳毛刷 猫狗 清洁"),
    ]
    records: List[ProductPair] = []
    for product_index, (english_name, english_caption, chinese_caption) in enumerate(base_products):
        image_vector = [0.0] * 8
        image_vector[product_index % 8] = 1.0
        noisy_vector = [value + random.uniform(-0.05, 0.05) for value in image_vector]
        records.append(
            ProductPair(
                record_id=f"pos-{product_index}",
                query_title=english_name,
                query_caption=chinese_caption,
                candidate_title=english_name.replace(" ", " "),
                candidate_caption=english_caption,
                query_image=image_vector,
                candidate_image=noisy_vector,
                label=1,
                ambiguity=0.15,
            )
        )
        negative_index = (product_index + 3) % len(base_products)
        negative_name, negative_caption, negative_chinese = base_products[negative_index]
        negative_vector = [0.0] * 8
        negative_vector[negative_index % 8] = 1.0
        records.append(
            ProductPair(
                record_id=f"neg-{product_index}",
                query_title=english_name,
                query_caption=chinese_caption,
                candidate_title=negative_name,
                candidate_caption=f"{negative_caption} {negative_chinese}",
                query_image=image_vector,
                candidate_image=negative_vector,
                label=0,
                ambiguity=0.10,
            )
        )
        if product_index % 2 == 0:
            near_vector = [(image_vector[value_index] + negative_vector[value_index]) / 2 for value_index in range(8)]
            records.append(
                ProductPair(
                    record_id=f"amb-{product_index}",
                    query_title=english_name,
                    query_caption=chinese_caption,
                    candidate_title=f"{english_name} compatible accessory",
                    candidate_caption=f"similar style {english_caption}",
                    query_image=image_vector,
                    candidate_image=near_vector,
                    label=1,
                    ambiguity=0.85,
                )
            )
    return records


def build_datasets(seed: int = 7, max_length: int = 32) -> tuple[ProductLinkingDataset, ProductLinkingDataset, Vocabulary]:
    records = build_toy_records(seed)
    texts = []
    for record in records:
        texts.append(record.query_text)
        texts.append(record.candidate_text)
    vocab = Vocabulary(texts)
    random.Random(seed).shuffle(records)
    split_index = max(1, int(len(records) * 0.75))
    train_records = records[:split_index]
    test_records = records[split_index:]
    return ProductLinkingDataset(train_records, vocab, max_length), ProductLinkingDataset(test_records, vocab, max_length), vocab
