from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class FactorSpec:
    colors: Tuple[str, ...] = ("red", "green", "blue", "purple")
    shapes: Tuple[str, ...] = ("circle", "square", "triangle", "star")
    styles: Tuple[str, ...] = ("photo", "sketch", "icon")


def _one_hot(index: int, size: int) -> np.ndarray:
    vec = np.zeros(size, dtype=np.float32)
    vec[index] = 1.0
    return vec


class ToyEmbeddingDataset(Dataset):
    """Synthetic embeddings with known factor structure.

    We create a ground-truth compositional embedding:
        z = A_color * onehot(color)
          + A_shape * onehot(shape)
          + A_style * onehot(style)
          + noise

    Then CEDAR tries to learn a rotation where the representation becomes sparse
    under a top-k bottleneck.
    """

    def __init__(
        self,
        num_samples: int = 8192,
        embedding_dim: int = 256,
        noise_std: float = 0.05,
        seed: int = 0,
    ) -> None:
        self.num_samples = int(num_samples)
        self.embedding_dim = int(embedding_dim)
        self.noise_std = float(noise_std)
        self.spec = FactorSpec()

        rng = np.random.default_rng(seed)

        self._labels = {
            "color": rng.integers(0, len(self.spec.colors), size=self.num_samples, endpoint=False),
            "shape": rng.integers(0, len(self.spec.shapes), size=self.num_samples, endpoint=False),
            "style": rng.integers(0, len(self.spec.styles), size=self.num_samples, endpoint=False),
        }

        # Create three random factor subspaces and normalize scale.
        def make_basis(num_factors: int) -> np.ndarray:
            mat = rng.normal(size=(num_factors, self.embedding_dim)).astype(np.float32)
            mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
            return mat

        self._A_color = make_basis(len(self.spec.colors))
        self._A_shape = make_basis(len(self.spec.shapes))
        self._A_style = make_basis(len(self.spec.styles))

        # Pre-generate embeddings for deterministic iteration.
        z = np.zeros((self.num_samples, self.embedding_dim), dtype=np.float32)
        for i in range(self.num_samples):
            c = int(self._labels["color"][i])
            s = int(self._labels["shape"][i])
            t = int(self._labels["style"][i])
            z[i] = (
                self._A_color[c]
                + self._A_shape[s]
                + self._A_style[t]
                + rng.normal(scale=self.noise_std, size=self.embedding_dim).astype(np.float32)
            )

        self._z = z

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        z = torch.from_numpy(self._z[idx])
        return {
            "z": z,
            "color": torch.tensor(int(self._labels["color"][idx]), dtype=torch.long),
            "shape": torch.tensor(int(self._labels["shape"][idx]), dtype=torch.long),
            "style": torch.tensor(int(self._labels["style"][idx]), dtype=torch.long),
        }


def build_factor_text_vocab(spec: FactorSpec) -> Dict[str, Tuple[str, ...]]:
    return {
        "color": spec.colors,
        "shape": spec.shapes,
        "style": spec.styles,
    }
