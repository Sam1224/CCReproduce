from dataclasses import dataclass
from typing import List

import torch
from torch.utils.data import Dataset


PRODUCTS = ["phone", "headphones", "laptop", "speaker", "monitor"]
ASPECTS = ["sim tray", "charging port", "power button", "bluetooth light", "hdmi port", "microphone"]
CONTEXTS = ["no sound", "not charging", "cannot receive calls", "screen flicker", "pairing failed"]
VOCAB = sorted(set(" ".join(PRODUCTS + ASPECTS + CONTEXTS).split()))
TOKEN_TO_ID = {token: index for index, token in enumerate(VOCAB)}


@dataclass
class MiragePair:
    chunk_meta: List[float]
    image_meta: List[float]
    label: float


def encode_metadata(product: str, aspect: str, context: str) -> List[float]:
    vector = torch.zeros(len(VOCAB) + len(PRODUCTS) + len(ASPECTS) + len(CONTEXTS))
    for token in f"{product} {aspect} {context}".split():
        vector[TOKEN_TO_ID[token]] = 1.0
    vector[len(VOCAB) + PRODUCTS.index(product)] = 1.0
    vector[len(VOCAB) + len(PRODUCTS) + ASPECTS.index(aspect)] = 1.0
    vector[len(VOCAB) + len(PRODUCTS) + len(ASPECTS) + CONTEXTS.index(context)] = 1.0
    return vector.tolist()


class MirageDataset(Dataset):
    def __init__(self):
        self.pairs: List[MiragePair] = []
        for product in PRODUCTS:
            for aspect in ASPECTS[:4]:
                for context in CONTEXTS[:3]:
                    chunk = encode_metadata(product, aspect, context)
                    image = encode_metadata(product, aspect, context)
                    self.pairs.append(MiragePair(chunk, image, 1.0))
                    wrong_aspect = ASPECTS[(ASPECTS.index(aspect) + 2) % len(ASPECTS)]
                    self.pairs.append(MiragePair(chunk, encode_metadata(product, wrong_aspect, context), 0.0))
                    wrong_product = PRODUCTS[(PRODUCTS.index(product) + 1) % len(PRODUCTS)]
                    self.pairs.append(MiragePair(chunk, encode_metadata(wrong_product, aspect, context), 0.0))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        pair = self.pairs[index]
        return {
            "chunk_meta": torch.tensor(pair.chunk_meta, dtype=torch.float32),
            "image_meta": torch.tensor(pair.image_meta, dtype=torch.float32),
            "label": torch.tensor(pair.label, dtype=torch.float32),
        }


def collate(batch: List[dict]) -> dict:
    return {
        "chunk_meta": torch.stack([item["chunk_meta"] for item in batch]),
        "image_meta": torch.stack([item["image_meta"] for item in batch]),
        "label": torch.stack([item["label"] for item in batch]),
    }
