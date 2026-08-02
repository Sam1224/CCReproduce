from dataclasses import dataclass
import torch
from torch.utils.data import Dataset


@dataclass
class SyntheticReTokenConfig:
    num_samples: int = 1024
    num_frames: int = 32
    image_dim: int = 64
    text_dim: int = 64
    relevant_frames: int = 2
    noise: float = 0.25
    seed: int = 7


class SyntheticVisualHaystackDataset(Dataset):
    def __init__(self, config: SyntheticReTokenConfig):
        self.config = config
        generator = torch.Generator().manual_seed(config.seed)
        self.questions = torch.randn(config.num_samples, config.text_dim, generator=generator)
        self.frames = torch.randn(config.num_samples, config.num_frames, config.image_dim, generator=generator) * config.noise
        self.labels = torch.zeros(config.num_samples, config.num_frames)
        for sample_index in range(config.num_samples):
            relevant = torch.randperm(config.num_frames, generator=generator)[: config.relevant_frames]
            self.labels[sample_index, relevant] = 1.0
            signal = self.questions[sample_index, : config.image_dim]
            self.frames[sample_index, relevant] += signal.unsqueeze(0)

    def __len__(self) -> int:
        return self.config.num_samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "frame_features": self.frames[index],
            "question_features": self.questions[index],
            "labels": self.labels[index],
        }
