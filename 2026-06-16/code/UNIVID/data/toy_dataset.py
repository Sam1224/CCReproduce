"""
Synthetic toy dataset for UNIVID evaluation.

Simulates short-video segments with:
  - Visual frame features (ViT-style)
  - ASR transcripts
  - Policy-aware ground-truth captions
  - Binary violation labels with fine-grained categories
"""

import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass, field
from typing import List, Optional, Dict


VIOLATION_CATEGORIES = [
    "safe",
    "explicit_content",
    "hate_speech",
    "violence",
    "spam_promotion",
    "misinformation",
    "copyright",
    "underage",
]

SAMPLE_TEXTS = [
    "welcome to my stream today we are doing a product review",
    "click the link in bio to get 50 percent off limited time offer",
    "this product cures all diseases doctors hate this trick",
    "showing you my favorite outfit haul from this brand",
    "violent scene incoming viewer discretion advised",
    "we love all communities this is a safe space for everyone",
    "buy cheap followers likes subscribers guaranteed delivery",
    "today we discuss the latest news in technology",
    "explicit language warning for mature audiences only",
    "this amazing supplement will make you lose weight fast",
]

POLICY_CAPTION_TEMPLATES = {
    "safe": "The content is compliant. Creator discusses {} in a neutral manner with no policy violations detected.",
    "explicit_content": "Policy violation detected: explicit content. The segment contains {} that violates community standards on explicit material.",
    "hate_speech": "Policy violation detected: hate speech. The segment contains {} targeting protected characteristics.",
    "spam_promotion": "Policy violation detected: spam/promotion. The segment promotes {} using deceptive or unsolicited commercial messaging.",
    "misinformation": "Policy violation detected: misinformation. The segment contains {} with false or misleading health/safety claims.",
    "violence": "Policy violation detected: violence. The segment depicts {} in violation of graphic content guidelines.",
    "copyright": "Policy violation detected: copyright. The segment uses {} without authorization.",
    "underage": "Policy violation detected: child safety. The segment involves {} that may endanger minors.",
}

FILL_INS = {
    "safe": ["product reviews", "lifestyle content", "educational material", "entertainment"],
    "explicit_content": ["suggestive imagery", "adult-oriented material"],
    "hate_speech": ["discriminatory language", "derogatory slurs"],
    "spam_promotion": ["unsolicited affiliate links", "fake giveaways"],
    "misinformation": ["unverified medical claims", "conspiracy theories"],
    "violence": ["graphic violent imagery", "combat scenes"],
    "copyright": ["copyrighted music", "licensed video clips"],
    "underage": ["minors in inappropriate situations"],
}


@dataclass
class VideoSegment:
    segment_id: str
    text: str
    visual_feat: torch.Tensor      # (visual_dim,) simulated ViT CLS features
    policy_caption: str
    label: int                      # 0=safe, 1=violation
    category: str
    risk_score: float               # ground-truth risk, used for filter calibration


class ToyVideoDataset(Dataset):
    """
    Toy dataset of short-video segments for UNIVID training/evaluation.

    Generates synthetic multimodal features with controlled violation rate.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        seed: int = 42,
        split: str = "train",
        violation_rate: float = 0.35,
        visual_dim: int = 768,
    ):
        rng = random.Random(seed)
        np_rng = np.random.RandomState(seed)
        torch.manual_seed(seed)

        self.samples: List[VideoSegment] = []
        self.visual_dim = visual_dim

        for i in range(num_samples):
            is_violation = rng.random() < violation_rate
            if is_violation:
                cat = rng.choice(VIOLATION_CATEGORIES[1:])
                label = 1
                risk_score = rng.uniform(0.55, 1.0)
            else:
                cat = "safe"
                label = 0
                risk_score = rng.uniform(0.0, 0.35)

            # Visual features: violation samples cluster differently
            visual_feat = torch.from_numpy(
                np_rng.randn(visual_dim).astype(np.float32)
            )
            if is_violation:
                violation_signal = np.zeros(visual_dim, dtype=np.float32)
                violation_signal[:32] = 2.0
                visual_feat += torch.from_numpy(violation_signal)
            visual_feat = torch.nn.functional.normalize(visual_feat, dim=0)

            text = rng.choice(SAMPLE_TEXTS)
            fill = rng.choice(FILL_INS[cat])
            policy_caption = POLICY_CAPTION_TEMPLATES[cat].format(fill)

            self.samples.append(VideoSegment(
                segment_id=f"{split}_{i:05d}",
                text=text,
                visual_feat=visual_feat,
                policy_caption=policy_caption,
                label=label,
                category=cat,
                risk_score=risk_score,
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seg = self.samples[idx]
        return {
            "segment_id": seg.segment_id,
            "text": seg.text,
            "visual_feat": seg.visual_feat,
            "policy_caption": seg.policy_caption,
            "label": seg.label,
            "category": seg.category,
            "risk_score": seg.risk_score,
        }

    def get_violation_samples(self) -> List[VideoSegment]:
        return [s for s in self.samples if s.label == 1]

    def get_reference_pool(self) -> Dict[str, List[torch.Tensor]]:
        """Return visual features of violation samples grouped by category."""
        pool: Dict[str, List[torch.Tensor]] = {}
        for s in self.samples:
            if s.label == 1:
                if s.category not in pool:
                    pool[s.category] = []
                pool[s.category].append(s.visual_feat)
        return pool


def get_dataloaders(
    batch_size: int = 32,
    num_samples: int = 1000,
    violation_rate: float = 0.35,
):
    train_ds = ToyVideoDataset(num_samples=num_samples, seed=42, split="train",
                               violation_rate=violation_rate)
    val_ds = ToyVideoDataset(num_samples=num_samples // 4, seed=123, split="val",
                             violation_rate=violation_rate)
    test_ds = ToyVideoDataset(num_samples=num_samples // 4, seed=456, split="test",
                              violation_rate=violation_rate)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
