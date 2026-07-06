from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import Dataset

RULES = [
    "absolute medical cure claim",
    "body shaping or height growth exaggeration",
    "financial return guarantee",
    "fake authority endorsement",
    "prohibited ingredient euphemism",
    "before-after misleading comparison",
]

CATEGORIES = ["compliant", "medical", "body", "finance", "authority", "ingredient", "comparison"]

TOY_SAMPLES = [
    {"text": "三天淡斑根治 老中医推荐 健康食品", "image_hint": [0.9, 0.1, 0.3, 0.2], "rules": [1, 0, 0, 1, 0, 0], "label": "medical"},
    {"text": "穿上立刻长高十厘米 明星同款鞋垫", "image_hint": [0.2, 0.9, 0.2, 0.4], "rules": [0, 1, 0, 1, 0, 0], "label": "body"},
    {"text": "稳赚不赔 月收益百分之二十 课程名额有限", "image_hint": [0.1, 0.1, 1.0, 0.2], "rules": [0, 0, 1, 0, 0, 0], "label": "finance"},
    {"text": "普通保温杯 不含治疗功效 支持七天无理由", "image_hint": [0.0, 0.0, 0.1, 0.1], "rules": [0, 0, 0, 0, 0, 0], "label": "compliant"},
    {"text": "清货价 不明粉末 代替敏感词描述燃脂胶囊", "image_hint": [0.6, 0.4, 0.2, 0.8], "rules": [0, 1, 0, 0, 1, 0], "label": "ingredient"},
    {"text": "使用前后对比图 七天瘦二十斤 无需运动", "image_hint": [0.5, 1.0, 0.1, 0.9], "rules": [0, 1, 0, 0, 0, 1], "label": "comparison"},
]


@dataclass
class Vocabulary:
    token_to_id: Dict[str, int]

    @classmethod
    def build(cls, texts: List[str]) -> "Vocabulary":
        token_to_id = {"<pad>": 0, "<unk>": 1}
        for text in texts:
            for char in text.lower():
                if char.strip() and char not in token_to_id:
                    token_to_id[char] = len(token_to_id)
        return cls(token_to_id)

    def encode(self, text: str, max_len: int) -> torch.Tensor:
        ids = [self.token_to_id.get(char, 1) for char in text.lower() if char.strip()][:max_len]
        ids += [0] * (max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)


class EvadeDataset(Dataset):
    def __init__(self, max_len: int = 48, vocab: Vocabulary | None = None):
        self.samples = TOY_SAMPLES
        self.max_len = max_len
        self.vocab = vocab or Vocabulary.build([sample["text"] for sample in self.samples])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = torch.tensor(sample["image_hint"], dtype=torch.float)
        rules = torch.tensor(sample["rules"], dtype=torch.float)
        return {
            "input_ids": self.vocab.encode(sample["text"], self.max_len),
            "image_features": image,
            "rule_targets": rules,
            "label": torch.tensor(CATEGORIES.index(sample["label"]), dtype=torch.long),
        }


def collate_batch(batch):
    return {key: torch.stack([item[key] for item in batch]) for key in batch[0]}
