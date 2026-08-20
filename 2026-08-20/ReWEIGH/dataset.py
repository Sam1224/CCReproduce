from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import Dataset


VOCAB = ["<pad>", "<bos>", "<eos>", "a", "shoe", "bag", "watch", "phone", "red", "blue", "green", "striped", "leather", "metal", "cotton", "wooden"]
TOKEN_TO_ID = {token: idx for idx, token in enumerate(VOCAB)}
OBJECT_TOKENS = {TOKEN_TO_ID[token] for token in ["shoe", "bag", "watch", "phone"]}


@dataclass
class ToyItem:
    image: torch.Tensor
    object_ids: List[int]
    caption_ids: List[int]


class ToyCommerceCaptionDataset(Dataset):
    def __init__(self, size: int = 128, seed: int = 7):
        generator = torch.Generator().manual_seed(seed)
        objects = ["shoe", "bag", "watch", "phone"]
        attrs = ["red", "blue", "green", "striped", "leather", "metal", "cotton", "wooden"]
        self.items: List[ToyItem] = []
        for idx in range(size):
            obj = objects[idx % len(objects)]
            attr = attrs[(idx * 3) % len(attrs)]
            image = torch.randn(12, generator=generator) * 0.15
            image[TOKEN_TO_ID[obj] % 12] += 2.2
            image[TOKEN_TO_ID[attr] % 12] += 1.4
            caption = [TOKEN_TO_ID["<bos>"], TOKEN_TO_ID["a"], TOKEN_TO_ID[attr], TOKEN_TO_ID[obj], TOKEN_TO_ID["<eos>"]]
            self.items.append(ToyItem(image=image, object_ids=[TOKEN_TO_ID[obj]], caption_ids=caption))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = self.items[index]
        return {
            "image": item.image,
            "caption_ids": torch.tensor(item.caption_ids, dtype=torch.long),
            "object_ids": torch.tensor(item.object_ids, dtype=torch.long),
        }


def collate_batch(batch):
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "caption_ids": torch.stack([item["caption_ids"] for item in batch]),
        "object_ids": torch.stack([item["object_ids"] for item in batch]),
    }
