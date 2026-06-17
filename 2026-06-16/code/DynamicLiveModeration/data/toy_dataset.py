"""
Toy multimodal livestream dataset for DynamicLiveModeration.

Generates synthetic multi-modal (text + "audio" features + "visual" features)
livestream segments with binary violation labels.

Interface matches the real dataset structure described in arXiv:2512.03553.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import json
import os


VIOLATION_CATEGORIES = [
    "hate_speech",
    "graphic_violence",
    "nudity",
    "dangerous_activity",
    "spam_scam",
    "copyright_infringement",
    "safe",  # negative class
]


class LivestreamSegment:
    """Single multimodal livestream segment."""
    def __init__(self, segment_id, text, audio_feat, visual_feat, label, category):
        self.segment_id = segment_id
        self.text = text          # ASR transcript
        self.audio_feat = audio_feat   # Audio embedding (512-d)
        self.visual_feat = visual_feat  # Visual embedding (768-d)
        self.label = label         # 1 = violation, 0 = safe
        self.category = category   # Violation category string


class ToyLivestreamDataset(Dataset):
    """
    Toy dataset simulating multimodal livestream segments.

    Paper: KDD 2026 | arXiv:2512.03553
    Modalities: text (ASR), audio features, visual features
    """

    TEXT_TEMPLATES_VIOLATION = [
        "Buy this now! Limited time offer only {}% off!",
        "This product cures all diseases guaranteed!",
        "Shocking content warning: {}",
        "Hack the system using this simple trick!",
        "Get rich quick scheme revealed: {}",
    ]
    TEXT_TEMPLATES_SAFE = [
        "Welcome to today's live show, let's review this product.",
        "Thanks for joining! Today we're talking about {}.",
        "Here's how to use this item step by step.",
        "Let me show you the features of this product.",
        "Great question from the audience about {}.",
    ]

    def __init__(self, num_samples=1000, violation_ratio=0.3, seed=42, split="train"):
        random.seed(seed)
        np.random.seed(seed)
        self.samples = []
        self.split = split

        num_violations = int(num_samples * violation_ratio)
        num_safe = num_samples - num_violations

        # Generate violation samples
        for i in range(num_violations):
            cat_idx = i % (len(VIOLATION_CATEGORIES) - 1)
            category = VIOLATION_CATEGORIES[cat_idx]
            text_template = random.choice(self.TEXT_TEMPLATES_VIOLATION)
            text = text_template.format(random.randint(10, 90))

            # Violation samples have distinct feature distribution
            audio_feat = torch.randn(512) + 1.0  # shifted distribution
            visual_feat = torch.randn(768) + 0.5

            self.samples.append(LivestreamSegment(
                segment_id=f"viol_{i}",
                text=text,
                audio_feat=audio_feat,
                visual_feat=visual_feat,
                label=1,
                category=category,
            ))

        # Generate safe samples
        for i in range(num_safe):
            text_template = random.choice(self.TEXT_TEMPLATES_SAFE)
            text = text_template.format(["products", "fashion", "tech", "food"][i % 4])

            audio_feat = torch.randn(512)
            visual_feat = torch.randn(768)

            self.samples.append(LivestreamSegment(
                segment_id=f"safe_{i}",
                text=text,
                audio_feat=audio_feat,
                visual_feat=visual_feat,
                label=0,
                category="safe",
            ))

        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seg = self.samples[idx]
        return {
            "text": seg.text,
            "audio_feat": seg.audio_feat,
            "visual_feat": seg.visual_feat,
            "label": torch.tensor(seg.label, dtype=torch.long),
            "category": seg.category,
        }

    def get_reference_bank(self, num_refs_per_category=5):
        """
        Build a reference bank of known violations.
        Used by the similarity pipeline (§3.2 of paper).
        Returns: dict of {category: [feature_vectors]}
        """
        ref_bank = {cat: [] for cat in VIOLATION_CATEGORIES[:-1]}
        for seg in self.samples:
            if seg.label == 1:
                cat = seg.category
                if len(ref_bank[cat]) < num_refs_per_category:
                    # Fuse modalities into single reference vector
                    fused = torch.cat([
                        seg.audio_feat[:256],
                        seg.visual_feat[:256],
                    ], dim=0)  # 512-d fused reference
                    ref_bank[cat].append(fused)
        return ref_bank


def get_dataloaders(batch_size=32, num_samples=1000):
    """Return train/val/test dataloaders."""
    train_ds = ToyLivestreamDataset(num_samples=num_samples, seed=42, split="train")
    val_ds = ToyLivestreamDataset(num_samples=num_samples // 5, seed=123, split="val")
    test_ds = ToyLivestreamDataset(num_samples=num_samples // 5, seed=456, split="test")

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )


if __name__ == "__main__":
    print("Generating toy livestream dataset...")
    ds = ToyLivestreamDataset(num_samples=200)
    print(f"  Total samples: {len(ds)}")
    sample = ds[0]
    print(f"  Sample keys: {list(sample.keys())}")
    print(f"  Audio feat shape: {sample['audio_feat'].shape}")
    print(f"  Visual feat shape: {sample['visual_feat'].shape}")
    print(f"  Label: {sample['label'].item()}")

    ref_bank = ds.get_reference_bank()
    print(f"  Reference bank categories: {list(ref_bank.keys())}")
    print("Toy dataset generation complete.")
