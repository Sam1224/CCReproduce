from dataclasses import dataclass
from typing import List

import torch
from torch.utils.data import Dataset


ATTRIBUTES = ["brand", "color", "material", "warning_label", "capacity", "style"]
TEXT_TOKENS = ["title", "category", "seller", "verified", "ocr", "image", "attribute", "json"] + ATTRIBUTES
TOKEN_TO_ID = {token: index for index, token in enumerate(TEXT_TOKENS)}


@dataclass
class ProductExample:
    image_features: List[float]
    text_tokens: List[int]
    target: List[float]


class EcomVLMDataset(Dataset):
    def __init__(self, size: int = 384, seed: int = 13):
        generator = torch.Generator().manual_seed(seed)
        self.examples: List[ProductExample] = []
        for _ in range(size):
            active = torch.randint(0, 2, (len(ATTRIBUTES),), generator=generator).float()
            noise = torch.randn(12, generator=generator) * 0.05
            image_features = torch.cat([active, active[:6] * 0.5 + noise[:6]]).tolist()
            text = [TOKEN_TO_ID["title"], TOKEN_TO_ID["category"], TOKEN_TO_ID["image"]]
            for attr_index, value in enumerate(active.tolist()):
                if value > 0.5:
                    text.append(TOKEN_TO_ID[ATTRIBUTES[attr_index]])
            while len(text) < 10:
                text.append(TOKEN_TO_ID["verified"])
            self.examples.append(ProductExample(image_features, text[:10], active.tolist()))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        example = self.examples[index]
        return {
            "image_features": torch.tensor(example.image_features, dtype=torch.float32),
            "text_tokens": torch.tensor(example.text_tokens, dtype=torch.long),
            "target": torch.tensor(example.target, dtype=torch.float32),
        }


def collate(batch: List[dict]) -> dict:
    return {
        "image_features": torch.stack([item["image_features"] for item in batch]),
        "text_tokens": torch.stack([item["text_tokens"] for item in batch]),
        "target": torch.stack([item["target"] for item in batch]),
    }
