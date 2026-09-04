from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import Dataset

ATTRS = ["red", "blue", "green", "yellow", "black", "white"]
OBJECTS = ["dress", "shoe", "bag", "hat", "chair", "plate"]
RELATIONS = ["left_of", "right_of", "above", "below"]

ATTR_TO_ID = {token: idx for idx, token in enumerate(ATTRS)}
OBJ_TO_ID = {token: idx for idx, token in enumerate(OBJECTS)}
REL_TO_ID = {token: idx for idx, token in enumerate(RELATIONS)}

PAD_ID = 0
ATTR_OFFSET = 1
OBJ_OFFSET = ATTR_OFFSET + len(ATTRS)
REL_OFFSET = OBJ_OFFSET + len(OBJECTS)
VOCAB_SIZE = REL_OFFSET + len(RELATIONS)
FEATURE_DIM = len(ATTRS) * 2 + len(OBJECTS) * 2 + len(RELATIONS)


@dataclass(frozen=True)
class Composition:
    attr1: str
    obj1: str
    relation: str
    attr2: str
    obj2: str


LEVEL_LABELS = {
    5: "full_match",
    4: "partial_presence",
    3: "attribute_error",
    2: "object_error",
    1: "total_error",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def random_composition(rng: random.Random) -> Composition:
    obj1, obj2 = rng.sample(OBJECTS, 2)
    attr1 = rng.choice(ATTRS)
    attr2 = rng.choice([item for item in ATTRS if item != attr1])
    relation = rng.choice(RELATIONS)
    return Composition(attr1=attr1, obj1=obj1, relation=relation, attr2=attr2, obj2=obj2)


def mutate_composition(base: Composition, level: int, rng: random.Random) -> Composition:
    if level == 5:
        return base
    if level == 4:
        relation = rng.choice([item for item in RELATIONS if item != base.relation])
        return Composition(base.attr1, base.obj1, relation, base.attr2, base.obj2)
    if level == 3:
        attr1 = rng.choice([item for item in ATTRS if item != base.attr1])
        return Composition(attr1, base.obj1, base.relation, base.attr2, base.obj2)
    if level == 2:
        obj2 = rng.choice([item for item in OBJECTS if item not in {base.obj1, base.obj2}])
        return Composition(base.attr1, base.obj1, base.relation, base.attr2, obj2)
    attr1 = rng.choice([item for item in ATTRS if item != base.attr1])
    obj1 = rng.choice([item for item in OBJECTS if item not in {base.obj1, base.obj2}])
    relation = rng.choice([item for item in RELATIONS if item != base.relation])
    attr2 = rng.choice([item for item in ATTRS if item != base.attr2])
    obj2 = rng.choice([item for item in OBJECTS if item not in {base.obj1, base.obj2, obj1}])
    return Composition(attr1, obj1, relation, attr2, obj2)


def encode_query(comp: Composition) -> List[int]:
    return [
        ATTR_TO_ID[comp.attr1] + ATTR_OFFSET,
        OBJ_TO_ID[comp.obj1] + OBJ_OFFSET,
        REL_TO_ID[comp.relation] + REL_OFFSET,
        ATTR_TO_ID[comp.attr2] + ATTR_OFFSET,
        OBJ_TO_ID[comp.obj2] + OBJ_OFFSET,
    ]


def encode_candidate(comp: Composition) -> torch.Tensor:
    feat = torch.zeros(FEATURE_DIM, dtype=torch.float32)
    feat[ATTR_TO_ID[comp.attr1]] = 1.0
    feat[len(ATTRS) + ATTR_TO_ID[comp.attr2]] = 1.0
    feat[len(ATTRS) * 2 + OBJ_TO_ID[comp.obj1]] = 1.0
    feat[len(ATTRS) * 2 + len(OBJECTS) + OBJ_TO_ID[comp.obj2]] = 1.0
    rel_index = len(ATTRS) * 2 + len(OBJECTS) * 2 + REL_TO_ID[comp.relation]
    feat[rel_index] = 1.0
    return feat


class CoresetDataset(Dataset):
    def __init__(self, split: str, size: int, seed: int = 42):
        self.split = split
        self.size = size
        self.seed = seed
        self.items = self._build_items()

    def _build_items(self) -> List[Dict[str, torch.Tensor]]:
        rng = random.Random(self.seed)
        items: List[Dict[str, torch.Tensor]] = []
        for _ in range(self.size):
            query_comp = random_composition(rng)
            query_tokens = torch.tensor(encode_query(query_comp), dtype=torch.long)
            candidates = []
            levels = []
            for level in [5, 4, 3, 2, 1]:
                cand = mutate_composition(query_comp, level=level, rng=rng)
                candidates.append(encode_candidate(cand))
                levels.append(level)
            candidate_tensor = torch.stack(candidates)
            teacher_scores = torch.tensor([5.0, 3.8, 2.7, 1.9, 1.0], dtype=torch.float32)
            items.append(
                {
                    "query_tokens": query_tokens,
                    "candidate_features": candidate_tensor,
                    "teacher_scores": teacher_scores,
                    "target_index": torch.tensor(0, dtype=torch.long),
                    "levels": torch.tensor(levels, dtype=torch.long),
                }
            )
        return items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.items[index]


def build_datasets(train_size: int = 640, val_size: int = 160, test_size: int = 200):
    return {
        "train": CoresetDataset("train", train_size, seed=11),
        "val": CoresetDataset("val", val_size, seed=17),
        "test": CoresetDataset("test", test_size, seed=23),
    }
