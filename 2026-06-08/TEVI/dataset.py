from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ToyConfig:
    num_samples: int = 4000
    num_concepts: int = 64
    embed_dim: int = 64
    concepts_per_caption: int = 6
    nuisance_per_image: int = 20
    noise_std: float = 0.02
    global_bias_strength: float = 0.0
    train_ratio: float = 0.8


class ToyMultimodalDataset(Dataset):
    """Synthetic (image_emb, text_emb) pairs.

    We explicitly create an "information imbalance":
    - image embedding = caption concepts + nuisance concepts + noise
    - text embedding  = caption concepts + small noise

    TEVI should learn to mask SAE latents to keep caption concepts and drop nuisance.
    """

    def __init__(self, cfg: ToyConfig, split: str, seed: int = 42):
        if split not in {"train", "test"}:
            raise ValueError("split must be train/test")

        self.cfg = cfg

        rng = np.random.default_rng(seed)
        if cfg.embed_dim == cfg.num_concepts:
            concept_vecs = np.eye(cfg.num_concepts, dtype=np.float32)
        else:
            concept_vecs = rng.normal(size=(cfg.num_concepts, cfg.embed_dim)).astype(np.float32)
            concept_vecs /= np.linalg.norm(concept_vecs, axis=1, keepdims=True) + 1e-8

        # choose caption concepts (multi-hot)
        caption_concepts = np.zeros((cfg.num_samples, cfg.num_concepts), dtype=np.float32)
        for i in range(cfg.num_samples):
            idx = rng.choice(cfg.num_concepts, size=cfg.concepts_per_caption, replace=False)
            caption_concepts[i, idx] = 1.0

        # choose nuisance concepts per image from remaining set
        nuisance_concepts = np.zeros((cfg.num_samples, cfg.num_concepts), dtype=np.float32)
        for i in range(cfg.num_samples):
            remaining = np.where(caption_concepts[i] == 0)[0]
            idx = rng.choice(remaining, size=cfg.nuisance_per_image, replace=False)
            nuisance_concepts[i, idx] = 1.0

        text_emb = caption_concepts @ concept_vecs
        text_emb += rng.normal(scale=cfg.noise_std, size=text_emb.shape).astype(np.float32)

        image_emb = (caption_concepts + nuisance_concepts) @ concept_vecs

        bias = rng.normal(size=(cfg.embed_dim,)).astype(np.float32)
        bias /= np.linalg.norm(bias) + 1e-8
        image_emb = image_emb + cfg.global_bias_strength * bias[None, :]

        image_emb += rng.normal(scale=cfg.noise_std, size=image_emb.shape).astype(np.float32)

        # L2 normalize as typical CLIP embedding practice
        def l2norm(x: np.ndarray) -> np.ndarray:
            return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

        text_emb = l2norm(text_emb)
        image_emb = l2norm(image_emb)

        n_train = int(cfg.num_samples * cfg.train_ratio)
        if split == "train":
            sl = slice(0, n_train)
        else:
            sl = slice(n_train, cfg.num_samples)

        self.image = torch.from_numpy(image_emb[sl])
        self.text = torch.from_numpy(text_emb[sl])

    def __len__(self) -> int:
        return self.image.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.image[idx], self.text[idx]
