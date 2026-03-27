from __future__ import annotations

import random
from typing import Dict, List

import torch
from torch.utils.data import Dataset


class ToyTrajectoryDataset(Dataset):
    def __init__(
        self,
        num_samples: int = 256,
        d_obs: int = 128,
        num_actions: int = 20,
        traj_len_range: tuple[int, int] = (6, 20),
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.d_obs = d_obs
        self.num_actions = num_actions
        self.traj_len_range = traj_len_range
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        t = self.rng.randint(*self.traj_len_range)
        obs = torch.randn(t, self.d_obs)
        actions = torch.randint(0, self.num_actions, (t,), dtype=torch.long)

        # "Success peer" logits are a sharpened distribution around the action.
        peer_logits = torch.full((t, self.num_actions), fill_value=-4.0)
        peer_logits.scatter_(1, actions.unsqueeze(1), 4.0)

        # Trajectory score correlated with how often action matches a simple rule.
        rule = (obs.mean(dim=-1) > 0).long() % self.num_actions
        correctness = (rule == actions).float().mean().item()
        traj_score = torch.tensor(correctness, dtype=torch.float32)

        return {"obs": obs, "actions": actions, "peer_logits": peer_logits, "traj_score": traj_score}


def collate_trajectories(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    lengths = torch.tensor([b["obs"].shape[0] for b in batch], dtype=torch.long)
    max_t = int(lengths.max().item())

    d_obs = batch[0]["obs"].shape[-1]
    num_actions = batch[0]["peer_logits"].shape[-1]

    obs = torch.zeros(len(batch), max_t, d_obs)
    actions = torch.full((len(batch), max_t), fill_value=-100, dtype=torch.long)
    peer_logits = torch.zeros(len(batch), max_t, num_actions)
    mask = torch.zeros(len(batch), max_t, dtype=torch.bool)
    traj_score = torch.stack([b["traj_score"] for b in batch], dim=0)

    for i, b in enumerate(batch):
        t = b["obs"].shape[0]
        obs[i, :t] = b["obs"]
        actions[i, :t] = b["actions"]
        peer_logits[i, :t] = b["peer_logits"]
        mask[i, :t] = True

    return {"obs": obs, "actions": actions, "peer_logits": peer_logits, "mask": mask, "traj_score": traj_score}
