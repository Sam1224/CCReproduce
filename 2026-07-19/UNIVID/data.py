import random
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


POLICIES = ["violence", "regulated_goods", "sexual_abuse", "integrity", "safe"]
CAPTION_VOCAB = ["safe", "weapon", "blood", "alcohol", "fraud", "minor", "context", "evidence"]


@dataclass
class VideoExample:
    frame_features: torch.Tensor
    text_features: torch.Tensor
    policy_id: int
    caption_targets: torch.Tensor
    violation_label: int
    leakage_label: int


class ToyVideoModerationDataset(Dataset):
    def __init__(self, size=512, frames=8, frame_dim=32, text_dim=16, seed=7):
        rng = np.random.default_rng(seed)
        self.examples = []
        policy_centers = rng.normal(size=(len(POLICIES), frame_dim)).astype("float32")
        text_centers = rng.normal(size=(len(POLICIES), text_dim)).astype("float32")
        for idx in range(size):
            policy_id = int(rng.integers(0, len(POLICIES)))
            risk_strength = float(rng.beta(2.0, 2.0))
            is_safe_policy = POLICIES[policy_id] == "safe"
            violation = int((risk_strength > 0.58) and not is_safe_policy)
            leakage = int(violation and risk_strength < 0.72)
            frames_arr = rng.normal(scale=0.45, size=(frames, frame_dim)).astype("float32")
            frames_arr += risk_strength * policy_centers[policy_id]
            text_arr = rng.normal(scale=0.35, size=(text_dim,)).astype("float32")
            text_arr += risk_strength * text_centers[policy_id]
            cap = np.zeros(len(CAPTION_VOCAB), dtype="float32")
            cap[0] = 1.0 - violation
            cap[(policy_id % (len(CAPTION_VOCAB) - 1)) + 1] = float(violation)
            cap[-1] = 1.0
            self.examples.append(
                VideoExample(
                    frame_features=torch.tensor(frames_arr),
                    text_features=torch.tensor(text_arr),
                    policy_id=policy_id,
                    caption_targets=torch.tensor(cap),
                    violation_label=violation,
                    leakage_label=leakage,
                )
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {
            "frame_features": ex.frame_features,
            "text_features": ex.text_features,
            "policy_id": torch.tensor(ex.policy_id, dtype=torch.long),
            "caption_targets": ex.caption_targets,
            "violation_label": torch.tensor(ex.violation_label, dtype=torch.float32),
            "leakage_label": torch.tensor(ex.leakage_label, dtype=torch.float32),
        }


def decode_caption(logits, threshold=0.5):
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    return [CAPTION_VOCAB[i] for i, p in enumerate(probs) if p >= threshold]
