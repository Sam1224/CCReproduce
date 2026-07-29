from typing import Dict, List

import torch
from torch.utils.data import Dataset


CONTENT_TYPES = ["album", "artist", "playlist", "show", "episode"]


class ToyShelfDataset(Dataset):
    def __init__(self, num_users: int = 512, catalogue_size: int = 2048, profile_dim: int = 40, embed_dim: int = 32, seed: int = 11):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.user_profiles = torch.randn(num_users, profile_dim, generator=generator)
        self.catalogue = torch.randn(catalogue_size, embed_dim, generator=generator)
        self.catalogue = torch.nn.functional.normalize(self.catalogue, dim=-1)
        self.catalogue_type = torch.randint(0, len(CONTENT_TYPES), (catalogue_size,), generator=generator)
        self.preference = torch.randn(num_users, embed_dim, generator=generator)
        self.preference = torch.nn.functional.normalize(self.preference, dim=-1)
        self.target_type = torch.randint(0, len(CONTENT_TYPES), (num_users,), generator=generator)
        self.positive_items = []
        for user_idx in range(num_users):
            type_mask = self.catalogue_type == self.target_type[user_idx]
            candidate_idx = type_mask.nonzero(as_tuple=False).flatten()
            scores = self.catalogue[candidate_idx] @ self.preference[user_idx]
            self.positive_items.append(candidate_idx[torch.topk(scores, 8).indices])

    def __len__(self) -> int:
        return self.user_profiles.size(0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "profile": self.user_profiles[idx].float(),
            "target_type": self.target_type[idx].long(),
            "positive_items": self.positive_items[idx].long(),
            "catalogue": self.catalogue.float(),
            "catalogue_type": self.catalogue_type.long(),
        }


def collate_shelves(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "profile": torch.stack([item["profile"] for item in batch]),
        "target_type": torch.stack([item["target_type"] for item in batch]),
        "positive_items": torch.stack([item["positive_items"] for item in batch]),
        "catalogue": batch[0]["catalogue"],
        "catalogue_type": batch[0]["catalogue_type"],
    }
