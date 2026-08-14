from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class CatalogSpec:
    text_dim: int = 12
    image_dim: int = 8
    attr_dim: int = 6


class PairDataset(Dataset):
    def __init__(self, payload: Dict[str, torch.Tensor]):
        self.payload = payload

    def __len__(self) -> int:
        return self.payload["label_line"].size(0)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {key: value[index] for key, value in self.payload.items()}


def build_catalog(num_groups: int = 240, seed: int = 42, spec: CatalogSpec | None = None) -> List[Dict[str, torch.Tensor]]:
    spec = spec or CatalogSpec()
    generator = torch.Generator().manual_seed(seed)
    items: List[Dict[str, torch.Tensor]] = []
    for group_index in range(num_groups):
        base_semantic = torch.randn(spec.text_dim, generator=generator)
        base_visual = torch.randn(spec.image_dim, generator=generator)
        parity_anchor = torch.randn(spec.attr_dim, generator=generator)
        for ladder_level in range(3):
            quality_vector = torch.zeros(spec.attr_dim)
            quality_vector[min(ladder_level, spec.attr_dim - 1)] = 1.0
            for variant_index in range(3):
                variant_vector = torch.zeros(spec.attr_dim)
                variant_vector[(variant_index + 3) % spec.attr_dim] = 1.0
                items.append(
                    {
                        "text": base_semantic + 0.05 * torch.randn(spec.text_dim, generator=generator),
                        "image": base_visual + 0.05 * torch.randn(spec.image_dim, generator=generator),
                        "attrs": parity_anchor + quality_vector + 0.3 * variant_vector,
                        "line_id": torch.tensor(group_index * 10 + ladder_level),
                        "ladder_id": torch.tensor(group_index),
                    }
                )
    return items


def build_pair_payload(seed: int = 42) -> Dict[str, torch.Tensor]:
    items = build_catalog(seed=seed)
    generator = torch.Generator().manual_seed(seed + 17)
    payload = {
        "left_text": [],
        "left_image": [],
        "left_attrs": [],
        "right_text": [],
        "right_image": [],
        "right_attrs": [],
        "label_line": [],
        "label_ladder": [],
    }

    line_buckets: Dict[int, List[int]] = {}
    ladder_buckets: Dict[int, List[int]] = {}
    for index, item in enumerate(items):
        line_buckets.setdefault(int(item["line_id"]), []).append(index)
        ladder_buckets.setdefault(int(item["ladder_id"]), []).append(index)

    line_keys = list(line_buckets.keys())
    ladder_keys = list(ladder_buckets.keys())

    def append_pair(left_item: Dict[str, torch.Tensor], right_item: Dict[str, torch.Tensor]) -> None:
        for key, prefix in (("text", "_text"), ("image", "_image"), ("attrs", "_attrs")):
            payload[f"left{prefix}"].append(left_item[key])
            payload[f"right{prefix}"].append(right_item[key])
        payload["label_line"].append(float(left_item["line_id"] == right_item["line_id"]))
        payload["label_ladder"].append(float(left_item["ladder_id"] == right_item["ladder_id"]))

    for _ in range(1536):
        ladder_key = ladder_keys[torch.randint(len(ladder_keys), (1,), generator=generator).item()]
        line_candidates = [k for k in line_keys if k // 10 == ladder_key]
        pos_line = line_candidates[torch.randint(len(line_candidates), (1,), generator=generator).item()]
        left_idx, right_idx = torch.randperm(len(line_buckets[pos_line]), generator=generator)[:2].tolist()
        append_pair(items[line_buckets[pos_line][left_idx]], items[line_buckets[pos_line][right_idx]])

        if len(line_candidates) > 1:
            first, second = torch.randperm(len(line_candidates), generator=generator)[:2].tolist()
            append_pair(items[line_buckets[line_candidates[first]][0]], items[line_buckets[line_candidates[second]][0]])

        negative_ladder = ladder_keys[torch.randint(len(ladder_keys), (1,), generator=generator).item()]
        while negative_ladder == ladder_key:
            negative_ladder = ladder_keys[torch.randint(len(ladder_keys), (1,), generator=generator).item()]
        append_pair(items[ladder_buckets[ladder_key][0]], items[ladder_buckets[negative_ladder][0]])

    return {key: torch.stack(value) if key.startswith("left") or key.startswith("right") else torch.tensor(value).float() for key, value in payload.items()}


def create_dataloaders(batch_size: int = 64, seed: int = 42) -> Tuple[DataLoader, DataLoader]:
    payload = build_pair_payload(seed=seed)
    split = int(0.8 * payload["label_line"].size(0))
    train_payload = {key: value[:split] for key, value in payload.items()}
    test_payload = {key: value[split:] for key, value in payload.items()}
    train_loader = DataLoader(PairDataset(train_payload), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(PairDataset(test_payload), batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
