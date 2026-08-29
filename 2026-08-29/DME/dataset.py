from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class VideoSearchExample:
    video_id: str
    title: str
    caption: str
    query: str
    frame_features: Sequence[Sequence[float]]
    label: int


class Vocab:
    def __init__(self, texts: Iterable[str]) -> None:
        self.token_to_id: Dict[str, int] = {"<pad>": 0, "<unk>": 1}
        for text in texts:
            for token in self.tokenize(text):
                if token not in self.token_to_id:
                    self.token_to_id[token] = len(self.token_to_id)

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text.lower())

    def encode(self, text: str, max_length: int) -> torch.Tensor:
        ids = [self.token_to_id.get(token, 1) for token in self.tokenize(text)[:max_length]]
        ids.extend([0] * (max_length - len(ids)))
        return torch.tensor(ids, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.token_to_id)


class VideoRetrievalDataset(Dataset):
    def __init__(self, examples: Sequence[VideoSearchExample], vocab: Vocab, max_length: int = 32) -> None:
        self.examples = list(examples)
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        example = self.examples[index]
        return {
            "video_id": example.video_id,
            "query_ids": self.vocab.encode(example.query, self.max_length),
            "text_ids": self.vocab.encode(f"{example.title} {example.caption}", self.max_length),
            "frames": torch.tensor(example.frame_features, dtype=torch.float32),
            "label": torch.tensor(float(example.label), dtype=torch.float32),
        }


def collate_examples(batch: Sequence[Dict[str, torch.Tensor | str]]) -> Dict[str, torch.Tensor | List[str]]:
    return {
        "video_id": [str(item["video_id"]) for item in batch],
        "query_ids": torch.stack([item["query_ids"] for item in batch]),
        "text_ids": torch.stack([item["text_ids"] for item in batch]),
        "frames": torch.stack([item["frames"] for item in batch]),
        "label": torch.stack([item["label"] for item in batch]),
    }


def build_toy_examples(seed: int = 13) -> List[VideoSearchExample]:
    random.seed(seed)
    topics = [
        ("beauty mirror live demo", "补光镜 直播 美妆 达人", "creator makeup lighting"),
        ("running shoe review", "跑鞋 测评 透气 运动", "sports sneaker review"),
        ("pet grooming tutorial", "宠物 梳毛 清洁 教程", "cat dog cleaning brush"),
        ("coffee mug unboxing", "陶瓷 咖啡杯 开箱", "office mug product"),
        ("wireless earbud comparison", "蓝牙 耳机 降噪 对比", "audio gadget review"),
        ("phone tripod shooting tips", "手机 三脚架 拍摄 技巧", "creator filming stand"),
    ]
    examples: List[VideoSearchExample] = []
    for index, (title, caption, query) in enumerate(topics):
        base = [0.0] * 10
        base[index % 10] = 1.0
        frames = [[value + random.uniform(-0.04, 0.04) for value in base] for _ in range(4)]
        examples.append(VideoSearchExample(f"pos-{index}", title, caption, query, frames, 1))
        negative_index = (index + 2) % len(topics)
        negative_title, negative_caption, _ = topics[negative_index]
        negative_base = [0.0] * 10
        negative_base[negative_index % 10] = 1.0
        negative_frames = [[value + random.uniform(-0.04, 0.04) for value in negative_base] for _ in range(4)]
        examples.append(VideoSearchExample(f"neg-{index}", negative_title, negative_caption, query, negative_frames, 0))
    return examples


def build_datasets(seed: int = 13) -> tuple[VideoRetrievalDataset, VideoRetrievalDataset, Vocab]:
    examples = build_toy_examples(seed)
    vocab = Vocab([text for example in examples for text in (example.query, example.title, example.caption)])
    random.Random(seed).shuffle(examples)
    split = int(len(examples) * 0.75)
    return VideoRetrievalDataset(examples[:split], vocab), VideoRetrievalDataset(examples[split:], vocab), vocab
