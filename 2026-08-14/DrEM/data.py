from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class DrEMBatchSpec:
    user_dim: int = 16
    item_dim: int = 24
    pxtr_dim: int = 7
    pairs_per_user: int = 6


class PairwiseRankingDataset(Dataset):
    def __init__(self, payload: Dict[str, torch.Tensor]):
        self.payload = payload

    def __len__(self) -> int:
        return self.payload["left_pxtr"].size(0)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {key: value[index] for key, value in self.payload.items()}


def _latent_reward(user_vec: torch.Tensor, item_vec: torch.Tensor, pxtr_true: torch.Tensor) -> torch.Tensor:
    collaborative = (user_vec * item_vec[:, : user_vec.size(-1)]).sum(dim=-1)
    behavior_signal = pxtr_true.sum(dim=-1)
    return collaborative + 0.8 * behavior_signal


def build_synthetic_pairs(
    num_users: int = 384,
    candidate_pool: int = 10,
    seed: int = 42,
    spec: DrEMBatchSpec | None = None,
) -> Dict[str, torch.Tensor]:
    spec = spec or DrEMBatchSpec()
    generator = torch.Generator().manual_seed(seed)

    user_vectors = torch.randn(num_users, spec.user_dim, generator=generator)
    item_vectors = torch.randn(num_users, candidate_pool, spec.item_dim, generator=generator)
    pxtr_true = torch.sigmoid(torch.randn(num_users, candidate_pool, spec.pxtr_dim, generator=generator))

    noise_scale = 0.04 + 0.16 * torch.rand(num_users, candidate_pool, spec.pxtr_dim, generator=generator)
    predicted_pxtr = torch.clamp(pxtr_true + noise_scale * torch.randn_like(pxtr_true, generator=generator), 0.0, 1.0)

    latent_reward = _latent_reward(
        user_vectors.unsqueeze(1).expand(-1, candidate_pool, -1).reshape(-1, spec.user_dim),
        item_vectors.reshape(-1, spec.item_dim),
        pxtr_true.reshape(-1, spec.pxtr_dim),
    ).reshape(num_users, candidate_pool)

    pair_payload = {
        "user": [],
        "left_item": [],
        "right_item": [],
        "left_pxtr": [],
        "right_pxtr": [],
        "left_noise": [],
        "right_noise": [],
        "label": [],
    }

    for user_index in range(num_users):
        for _ in range(spec.pairs_per_user):
            left, right = torch.randperm(candidate_pool, generator=generator)[:2].tolist()
            pair_payload["user"].append(user_vectors[user_index])
            pair_payload["left_item"].append(item_vectors[user_index, left])
            pair_payload["right_item"].append(item_vectors[user_index, right])
            pair_payload["left_pxtr"].append(predicted_pxtr[user_index, left])
            pair_payload["right_pxtr"].append(predicted_pxtr[user_index, right])
            pair_payload["left_noise"].append(noise_scale[user_index, left])
            pair_payload["right_noise"].append(noise_scale[user_index, right])
            pair_payload["label"].append(float(latent_reward[user_index, left] > latent_reward[user_index, right]))

    return {key: torch.stack(value) if key != "label" else torch.tensor(value).float() for key, value in pair_payload.items()}


def create_dataloaders(batch_size: int = 64, seed: int = 42) -> Tuple[DataLoader, DataLoader]:
    payload = build_synthetic_pairs(seed=seed)
    split = int(0.8 * payload["label"].size(0))
    train_payload = {key: value[:split] for key, value in payload.items()}
    test_payload = {key: value[split:] for key, value in payload.items()}
    train_loader = DataLoader(PairwiseRankingDataset(train_payload), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(PairwiseRankingDataset(test_payload), batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
