from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import Dataset


@dataclass
class ToyAgentConfig:
    num_samples: int = 512
    turns: int = 6
    tokens_per_turn: int = 8
    vocab_size: int = 128
    num_actions: int = 6
    seed: int = 7


class ToyAgentTrajectoryDataset(Dataset):
    def __init__(self, config: ToyAgentConfig = ToyAgentConfig()):
        self.config = config
        generator = torch.Generator().manual_seed(config.seed)
        self.samples: List[Dict[str, torch.Tensor]] = []
        for _ in range(config.num_samples):
            observations = torch.randint(
                low=4,
                high=config.vocab_size,
                size=(config.turns, config.tokens_per_turn),
                generator=generator,
            )
            pivotal_turn = torch.randint(0, config.turns, (1,), generator=generator).item()
            target_actions = observations[:, 0] % config.num_actions
            distractor_actions = (target_actions + torch.randint(1, config.num_actions, (config.turns,), generator=generator)) % config.num_actions
            actions = target_actions.clone()
            error_mask = torch.rand(config.turns, generator=generator) < 0.25
            actions[error_mask] = distractor_actions[error_mask]
            reward = torch.tensor(float(actions[pivotal_turn] == target_actions[pivotal_turn]))

            teacher_log_probs = torch.full((config.turns, config.tokens_per_turn), -0.50)
            student_log_probs = torch.full((config.turns, config.tokens_per_turn), -0.50)
            teacher_log_probs[pivotal_turn] = -0.03 if reward.item() == 1 else -1.20
            student_log_probs[pivotal_turn] = -1.20 if reward.item() == 1 else -0.03
            token_mask = torch.ones(config.turns, config.tokens_per_turn)

            self.samples.append(
                {
                    "observations": observations.long(),
                    "actions": actions.long(),
                    "target_actions": target_actions.long(),
                    "teacher_log_probs": teacher_log_probs.float(),
                    "student_log_probs": student_log_probs.float(),
                    "token_mask": token_mask.float(),
                    "reward": reward.float(),
                    "pivotal_turn": torch.tensor(pivotal_turn).long(),
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.samples[index]


def collate_trajectories(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {key: torch.stack([item[key] for item in batch]) for key in batch[0]}
