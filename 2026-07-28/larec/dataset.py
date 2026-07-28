import json
import os
from typing import Dict, List

import torch
from torch.utils.data import Dataset


class LaRecDataset(Dataset):
    def __init__(self, split: str = "train", data_path: str = None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        resolved_path = data_path or os.path.join(base_dir, "toy_data.json")
        with open(resolved_path, "r", encoding="utf-8") as file:
            payload = json.load(file)

        self.items = payload["items"]
        self.item_map = {item["item_id"]: item for item in self.items}
        self.examples = [example for example in payload["examples"] if example["split"] == split]
        self.num_items = max(item["item_id"] for item in self.items) + 1

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        example = self.examples[index]
        return {
            "user_id": example["user_id"],
            "history_item_ids": example["history_item_ids"],
            "step_item_ids": example["step_item_ids"],
            "interest_item_ids": example["interest_item_ids"],
            "target_item_id": example["target_item_id"],
            "negative_item_ids": example["negative_item_ids"],
        }


def _pad_2d(sequences: List[List[int]], pad_value: int = 0) -> torch.Tensor:
    max_len = max(len(sequence) for sequence in sequences)
    padded = [sequence + [pad_value] * (max_len - len(sequence)) for sequence in sequences]
    return torch.tensor(padded, dtype=torch.long)


def collate_fn(batch: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
    history = _pad_2d([item["history_item_ids"] for item in batch])
    steps = _pad_2d([item["step_item_ids"] for item in batch])
    interests = _pad_2d([item["interest_item_ids"] for item in batch])
    negatives = _pad_2d([item["negative_item_ids"] for item in batch])
    targets = torch.tensor([item["target_item_id"] for item in batch], dtype=torch.long)
    users = torch.tensor([item["user_id"] for item in batch], dtype=torch.long)

    history_mask = (history != 0).float()
    step_mask = (steps != 0).float()
    interest_mask = (interests != 0).float()
    negative_mask = (negatives != 0).float()

    return {
        "user_id": users,
        "history_item_ids": history,
        "history_mask": history_mask,
        "step_item_ids": steps,
        "step_mask": step_mask,
        "interest_item_ids": interests,
        "interest_mask": interest_mask,
        "target_item_id": targets,
        "negative_item_ids": negatives,
        "negative_mask": negative_mask,
    }
