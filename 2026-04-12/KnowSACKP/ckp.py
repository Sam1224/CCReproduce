from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch

from data import ToyRecDataset
from model import MockLLM


@dataclass
class CKPCache:
    knowledge_score: np.ndarray  # [items]
    augment_mask: np.ndarray  # bool[items]


def compute_ckp_scores(
    *,
    dataset: ToyRecDataset,
    llm: MockLLM,
    sample_users: int = 500,
    pair_samples: int = 8,
    seed: int = 42,
) -> np.ndarray:
    """Compute a toy CKP-like knowledge score per item.

    For each item i, we estimate how often the model can correctly compare i
    against another candidate j, conditioned on a user's history.

    Oracle preference is defined by ground-truth latent similarity to the user's latent profile.
    Model preference is defined by the MockLLM parametric representations.
    """

    rng = np.random.default_rng(seed)
    item_count = len(dataset.item_tokens)

    scores = np.zeros(item_count, dtype=np.float32)
    counts = np.zeros(item_count, dtype=np.int32)

    user_indices = rng.choice(len(dataset.user_cases), size=min(sample_users, len(dataset.user_cases)), replace=False)

    latent = dataset.item_latent[:, : llm.dim]

    for ui in user_indices:
        case = dataset.user_cases[int(ui)]
        hist = np.asarray(case.history, dtype=np.int64)
        user_oracle = latent[hist].mean(axis=0)
        user_oracle = user_oracle / (np.linalg.norm(user_oracle) + 1e-12)

        # Model-side user vector uses parametric representations.
        with torch.no_grad():
            hist_t = torch.tensor(case.history, dtype=torch.long, device=llm.device)
            user_model = llm.param_emb[hist_t].mean(dim=0)
            user_model = user_model / (user_model.norm() + 1e-12)

        cand = case.candidates
        for _ in range(pair_samples):
            i, j = rng.choice(cand, size=2, replace=False)
            i = int(i)
            j = int(j)

            oracle_pref = float(user_oracle @ latent[i]) > float(user_oracle @ latent[j])
            with torch.no_grad():
                i_score = (user_model * llm.param_emb[i]).sum().item()
                j_score = (user_model * llm.param_emb[j]).sum().item()
            model_pref = i_score > j_score

            correct = 1.0 if oracle_pref == model_pref else 0.0
            scores[i] += correct
            scores[j] += correct
            counts[i] += 1
            counts[j] += 1

    counts = np.maximum(counts, 1)
    return scores / counts


def build_selective_mask(knowledge_score: np.ndarray, *, augment_ratio: float = 0.3) -> np.ndarray:
    thresh = np.quantile(knowledge_score, augment_ratio)
    return knowledge_score <= thresh
