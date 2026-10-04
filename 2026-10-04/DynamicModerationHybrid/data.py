import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class FeatureConfig:
    visual_dim: int = 64
    audio_dim: int = 32
    text_dim: int = 48
    num_classes: int = 5
    num_references: int = 24
    seed: int = 7


class ToyLivestreamDataset(Dataset):
    def __init__(self, size: int = 2000, config: FeatureConfig | None = None):
        self.config = config or FeatureConfig()
        rng = np.random.default_rng(self.config.seed)
        self.class_centers = rng.normal(size=(self.config.num_classes, self.feature_dim)).astype("float32")
        self.samples = []
        for index in range(size):
            label = int(rng.integers(0, self.config.num_classes))
            violation_strength = rng.uniform(0.25, 1.0) if label > 0 else rng.uniform(0.0, 0.35)
            feature = self.class_centers[label] * violation_strength + rng.normal(scale=0.65, size=self.feature_dim)
            visual, audio, text = self._split(feature.astype("float32"))
            reference_id = int(label * max(1, self.config.num_references // self.config.num_classes) + rng.integers(0, 3))
            self.samples.append((visual, audio, text, label, reference_id))

    @property
    def feature_dim(self) -> int:
        return self.config.visual_dim + self.config.audio_dim + self.config.text_dim

    def _split(self, feature: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        v_end = self.config.visual_dim
        a_end = v_end + self.config.audio_dim
        return feature[:v_end], feature[v_end:a_end], feature[a_end:]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        visual, audio, text, label, reference_id = self.samples[index]
        return {
            "visual": torch.tensor(visual),
            "audio": torch.tensor(audio),
            "text": torch.tensor(text),
            "label": torch.tensor(label, dtype=torch.long),
            "reference_id": torch.tensor(reference_id, dtype=torch.long),
        }


def build_reference_bank(dataset: ToyLivestreamDataset, embedding_dim: int = 128) -> Dict[str, torch.Tensor]:
    rng = np.random.default_rng(dataset.config.seed + 13)
    raw_refs = rng.normal(size=(dataset.config.num_references, dataset.feature_dim)).astype("float32")
    for ref_id in range(dataset.config.num_references):
        class_id = min(dataset.config.num_classes - 1, ref_id // max(1, dataset.config.num_references // dataset.config.num_classes))
        raw_refs[ref_id] += dataset.class_centers[class_id]
    return {"raw": torch.tensor(raw_refs), "ids": torch.arange(dataset.config.num_references)}
