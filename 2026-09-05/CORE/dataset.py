from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


ATTRIBUTES = [
    "red",
    "blue",
    "green",
    "striped",
    "leather",
    "metal",
    "wooden",
    "small",
    "large",
    "round",
]

OBJECTS = [
    "dress",
    "chair",
    "phone",
    "bag",
    "cup",
    "shoe",
    "lamp",
    "watch",
    "sofa",
    "plate",
]

RELATIONS = ["beside", "behind", "above", "under", "inside", "near"]

LEVEL_SCORES = {
    5: 1.00,
    4: 0.78,
    3: 0.55,
    2: 0.32,
    1: 0.05,
}


@dataclass(frozen=True)
class Sample:
    query_tokens: np.ndarray
    candidate_tokens: np.ndarray
    image_features: np.ndarray
    level: int
    teacher_score: float
    group_id: int


def build_vocab() -> Dict[str, int]:
    words = ["[PAD]", "[IMG]", "a", "and", "is", "the", "product", "content"]
    words += ATTRIBUTES + OBJECTS + RELATIONS
    return {word: idx for idx, word in enumerate(words)}


def encode(tokens: List[str], vocab: Dict[str, int], max_len: int) -> np.ndarray:
    ids = [vocab.get(tok, 0) for tok in tokens[:max_len]]
    ids += [0] * (max_len - len(ids))
    return np.asarray(ids, dtype=np.int64)


def _different(items: List[str], value: str, rng: np.random.Generator) -> str:
    pool = [item for item in items if item != value]
    return str(rng.choice(pool))


def _compose_text(attr_a: str, obj_a: str, rel: str, attr_b: str, obj_b: str) -> List[str]:
    return ["a", attr_a, obj_a, rel, "a", attr_b, obj_b]


class CompositionalListDataset(Dataset):
    def __init__(
        self,
        num_groups: int = 512,
        max_len: int = 10,
        image_dim: int = 32,
        seed: int = 7,
    ) -> None:
        self.vocab = build_vocab()
        self.max_len = max_len
        self.image_dim = image_dim
        self.samples: List[Sample] = []
        rng = np.random.default_rng(seed)
        concept_vectors = {
            token: rng.normal(size=image_dim).astype(np.float32)
            for token in ATTRIBUTES + OBJECTS + RELATIONS
        }

        for group_id in range(num_groups):
            attr_a = str(rng.choice(ATTRIBUTES))
            obj_a = str(rng.choice(OBJECTS))
            attr_b = _different(ATTRIBUTES, attr_a, rng)
            obj_b = _different(OBJECTS, obj_a, rng)
            rel = str(rng.choice(RELATIONS))
            query = _compose_text(attr_a, obj_a, rel, attr_b, obj_b)

            candidates: List[Tuple[int, List[str]]] = [
                (5, _compose_text(attr_a, obj_a, rel, attr_b, obj_b)),
                (4, _compose_text(attr_a, obj_a, rel, _different(ATTRIBUTES, attr_b, rng), obj_b)),
                (3, _compose_text(_different(ATTRIBUTES, attr_a, rng), obj_a, rel, attr_b, obj_b)),
                (2, _compose_text(attr_a, _different(OBJECTS, obj_a, rng), rel, attr_b, obj_b)),
                (1, _compose_text(_different(ATTRIBUTES, attr_a, rng), _different(OBJECTS, obj_a, rng), str(rng.choice(RELATIONS)), _different(ATTRIBUTES, attr_b, rng), _different(OBJECTS, obj_b, rng))),
            ]
            rng.shuffle(candidates)

            for level, cand in candidates:
                vec = np.zeros(image_dim, dtype=np.float32)
                for token in cand:
                    if token in concept_vectors:
                        vec += concept_vectors[token]
                vec += rng.normal(scale=0.08, size=image_dim).astype(np.float32)
                self.samples.append(
                    Sample(
                        query_tokens=encode(query, self.vocab, max_len),
                        candidate_tokens=encode(cand, self.vocab, max_len),
                        image_features=vec,
                        level=level,
                        teacher_score=LEVEL_SCORES[level] + float(rng.normal(scale=0.015)),
                        group_id=group_id,
                    )
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Sample:
        return self.samples[idx]


def split_by_group(dataset: CompositionalListDataset, test_ratio: float = 0.2) -> Tuple[List[int], List[int]]:
    groups = sorted({sample.group_id for sample in dataset.samples})
    cutoff = int(len(groups) * (1.0 - test_ratio))
    train_groups = set(groups[:cutoff])
    train_idx, test_idx = [], []
    for idx, sample in enumerate(dataset.samples):
        if sample.group_id in train_groups:
            train_idx.append(idx)
        else:
            test_idx.append(idx)
    return train_idx, test_idx


def collate_samples(rows: List[Sample]) -> Dict[str, torch.Tensor]:
    return {
        "query_tokens": torch.tensor(np.stack([row.query_tokens for row in rows]), dtype=torch.long),
        "candidate_tokens": torch.tensor(np.stack([row.candidate_tokens for row in rows]), dtype=torch.long),
        "image_features": torch.tensor(np.stack([row.image_features for row in rows]), dtype=torch.float32),
        "level": torch.tensor([row.level for row in rows], dtype=torch.long),
        "teacher_score": torch.tensor([row.teacher_score for row in rows], dtype=torch.float32),
        "group_id": torch.tensor([row.group_id for row in rows], dtype=torch.long),
    }
