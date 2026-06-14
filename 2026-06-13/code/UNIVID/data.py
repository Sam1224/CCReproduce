"""
Toy data pipeline for UNIVID reproduction.
Generates synthetic video moderation data with policy-aware captions.

Paper: UNIVID: Unified Vision-Language Model for Video Moderation (arXiv:2606.05748)
Authors: Kejuan Yang et al., ByteDance
"""

import random
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
import torch
from torch.utils.data import Dataset

# ── Policy categories (simplified from paper's multi-policy setting) ──────────
POLICY_CATEGORIES = [
    "violence",
    "nudity",
    "hate_speech",
    "spam",
    "misinformation",
    "copyright_violation",
    "safe",
]

# ── Synthetic caption templates per policy ────────────────────────────────────
CAPTION_TEMPLATES = {
    "violence": [
        "The video shows {subject} engaged in {action} with visible {element}.",
        "Content depicts {subject} performing {action}, resulting in {outcome}.",
    ],
    "nudity": [
        "The video contains {subject} with {attribute} visible in {context}.",
        "Content shows {subject} in {context} with {attribute} exposure.",
    ],
    "hate_speech": [
        "The video includes {subject} making {type} remarks about {target}.",
        "Audio/text contains {type} language targeting {target} group.",
    ],
    "spam": [
        "The video repeatedly promotes {product} with {tactic} tactics.",
        "Content contains {count} promotional insertions for {product}.",
    ],
    "misinformation": [
        "The video claims {claim} without credible {evidence}.",
        "Content presents {claim} as fact, contradicting {source}.",
    ],
    "copyright_violation": [
        "The video contains {duration} seconds of copyrighted {media}.",
        "Content uses {media} without proper {license} authorization.",
    ],
    "safe": [
        "The video shows {subject} in a {setting} engaged in {activity}.",
        "Content depicts {subject} sharing {content_type} in a {setting}.",
    ],
}

FILL_WORDS = {
    "subject": ["a person", "the creator", "multiple individuals", "the host"],
    "action": ["physical altercation", "aggressive behavior", "combat"],
    "element": ["blood", "weapons", "injuries"],
    "outcome": ["visible harm", "physical damage"],
    "attribute": ["explicit content", "inappropriate material"],
    "context": ["a private setting", "a public space", "a live stream"],
    "type": ["discriminatory", "hateful", "offensive"],
    "target": ["ethnic", "religious", "gender"],
    "product": ["supplements", "electronics", "fashion items"],
    "tactic": ["misleading", "repetitive", "fake-scarcity"],
    "count": ["5", "8", "12"],
    "claim": ["miraculous health benefits", "false political statements"],
    "evidence": ["scientific backing", "reliable sources"],
    "source": ["medical consensus", "established facts"],
    "duration": ["30", "45", "60"],
    "media": ["music", "movie clips", "TV footage"],
    "license": ["licensing", "attribution", "permission"],
    "setting": ["home", "outdoor environment", "studio"],
    "activity": ["cooking", "exercising", "creating content"],
    "content_type": ["educational content", "entertainment", "lifestyle tips"],
}


def generate_caption(policy: str) -> str:
    templates = CAPTION_TEMPLATES[policy]
    template = random.choice(templates)
    result = template
    for key, options in FILL_WORDS.items():
        placeholder = "{" + key + "}"
        if placeholder in result:
            result = result.replace(placeholder, random.choice(options), 1)
    return result


@dataclass
class VideoSample:
    video_id: str
    policy_label: str
    is_violation: bool
    caption: str
    visual_features: Optional[torch.Tensor] = None

    def to_dict(self):
        return {
            "video_id": self.video_id,
            "policy_label": self.policy_label,
            "is_violation": self.is_violation,
            "caption": self.caption,
        }


def generate_synthetic_dataset(
    n_samples: int = 1000,
    violation_ratio: float = 0.3,
    seed: int = 42,
) -> List[VideoSample]:
    random.seed(seed)
    samples = []
    violation_policies = [p for p in POLICY_CATEGORIES if p != "safe"]

    for i in range(n_samples):
        is_violation = random.random() < violation_ratio
        if is_violation:
            policy = random.choice(violation_policies)
        else:
            policy = "safe"

        caption = generate_caption(policy)
        sample = VideoSample(
            video_id=f"vid_{i:06d}",
            policy_label=policy,
            is_violation=is_violation,
            caption=caption,
        )
        samples.append(sample)

    return samples


def save_dataset(samples: List[VideoSample], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([s.to_dict() for s in samples], f, indent=2)
    print(f"Saved {len(samples)} samples to {path}")


def load_dataset(path: str) -> List[VideoSample]:
    with open(path) as f:
        data = json.load(f)
    return [VideoSample(**d) for d in data]


class ModerationDataset(Dataset):
    """
    PyTorch Dataset wrapping VideoSample list.
    In the real paper, visual_features come from a frozen ViT encoder over
    sampled video frames; here we use random tensors as placeholders.
    """

    def __init__(
        self,
        samples: List[VideoSample],
        tokenizer,
        max_caption_length: int = 128,
        visual_dim: int = 768,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_caption_length = max_caption_length
        self.visual_dim = visual_dim
        self.label2id = {p: i for i, p in enumerate(POLICY_CATEGORIES)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Tokenize policy-aware caption
        encoding = self.tokenizer(
            sample.caption,
            max_length=self.max_caption_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Toy visual features: random tensor simulating ViT pooled output
        # In the real pipeline these come from a vision encoder over video frames
        visual_features = torch.randn(self.visual_dim)

        label = self.label2id[sample.policy_label]
        violation = int(sample.is_violation)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "visual_features": visual_features,
            "policy_label": torch.tensor(label, dtype=torch.long),
            "violation_label": torch.tensor(violation, dtype=torch.float),
            "caption": sample.caption,
            "video_id": sample.video_id,
        }


if __name__ == "__main__":
    samples = generate_synthetic_dataset(n_samples=2000)
    train = samples[:1600]
    val = samples[1600:]
    save_dataset(train, "data/train.json")
    save_dataset(val, "data/val.json")
    print(f"Train: {len(train)}, Val: {len(val)}")
    print("Sample:", train[0].to_dict())
