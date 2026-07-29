import random
from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import Dataset


@dataclass
class StreamSample:
    content: torch.Tensor
    author_id: int
    target_users: torch.Tensor
    non_target_users: torch.Tensor
    positive_item: int
    item_bank: torch.Tensor


class ToyLiveStreamDataset(Dataset):
    def __init__(self, num_items: int = 512, num_users: int = 2048, content_dim: int = 48, embed_dim: int = 32, users_per_side: int = 16, seed: int = 7):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.content = torch.randn(num_items, content_dim, generator=generator)
        self.item_semantics = torch.randn(num_items, embed_dim, generator=generator)
        self.author_ids = torch.randint(0, max(8, num_items // 8), (num_items,), generator=generator)
        self.user_embeddings = torch.randn(num_users, embed_dim, generator=generator)
        self.user_embeddings = torch.nn.functional.normalize(self.user_embeddings, dim=-1)
        noise = torch.randn(num_items, embed_dim, generator=generator)
        self.item_bank = torch.nn.functional.normalize(self.item_semantics + 0.15 * noise, dim=-1)
        self.users_per_side = users_per_side
        self.seed = seed

    def __len__(self) -> int:
        return self.content.size(0)

    def _sample_users(self, item_idx: int) -> Dict[str, torch.Tensor]:
        scores = self.user_embeddings @ self.item_semantics[item_idx]
        target = torch.topk(scores, self.users_per_side).indices
        non_target = torch.topk(-scores, self.users_per_side).indices
        return {
            "target": self.user_embeddings[target],
            "non_target": self.user_embeddings[non_target],
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        user_sets = self._sample_users(idx)
        return {
            "content": self.content[idx].float(),
            "author_id": torch.tensor(self.author_ids[idx]).long(),
            "target_users": user_sets["target"].float(),
            "non_target_users": user_sets["non_target"].float(),
            "positive_item": torch.tensor(idx).long(),
            "item_bank": self.item_bank.float(),
        }


def collate_streams(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "content": torch.stack([item["content"] for item in batch]),
        "author_id": torch.stack([item["author_id"] for item in batch]),
        "target_users": torch.stack([item["target_users"] for item in batch]),
        "non_target_users": torch.stack([item["non_target_users"] for item in batch]),
        "positive_item": torch.stack([item["positive_item"] for item in batch]),
        "item_bank": batch[0]["item_bank"],
    }
