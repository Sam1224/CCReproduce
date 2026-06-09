"""
UNIVID — Dataset & Data Interface

Toy dataset implementation aligned with the paper's data interface.
In production, ByteDance uses proprietary labeled video datasets with
violation annotations + policy documents. This provides a compatible interface.
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from typing import List, Dict, Optional, Tuple


class VideoModerationDataset(Dataset):
    """
    Dataset for UNIVID policy-aware caption training.

    Expected data format (JSONL):
    {
        "video_id": "vid_001",
        "frames": ["frame_001.jpg", ...],   # extracted frame paths
        "policy_text": "Policy: Sexual content includes...",
        "caption": "The video shows a person dancing...",  # target
        "violation_labels": [0, 1, 0, 0, ...],  # multi-label violation types
        "is_violation": true
    }
    """

    VIOLATION_TYPES = [
        "sexual_content",
        "violence",
        "hate_speech",
        "spam",
        "misinformation",
        "copyright",
        "dangerous_acts",
        "minor_safety",
        "political_content",
        "other",
    ]

    def __init__(
        self,
        data_path: str,
        tokenizer,
        image_processor,
        num_frames: int = 8,
        max_policy_len: int = 128,
        max_caption_len: int = 256,
        split: str = "train",
    ):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.num_frames = num_frames
        self.max_policy_len = max_policy_len
        self.max_caption_len = max_caption_len

        data_file = os.path.join(data_path, f"{split}.jsonl")
        if os.path.exists(data_file):
            with open(data_file) as f:
                self.samples = [json.loads(line) for line in f]
        else:
            # Generate synthetic toy data for testing
            self.samples = self._generate_toy_data(n=200 if split == "train" else 50)

    def _generate_toy_data(self, n: int) -> List[Dict]:
        """Generate synthetic data for testing the pipeline."""
        samples = []
        policies = [
            "Policy: Detect sexually explicit content including nudity and suggestive behavior.",
            "Policy: Detect violent content including gore, fighting, and dangerous acts.",
            "Policy: Detect hate speech targeting individuals based on race, gender, or religion.",
            "Policy: Detect spam and deceptive promotional content.",
        ]
        captions_safe = [
            "The video shows a person cooking a meal in their kitchen.",
            "A cat playing with a toy mouse on a carpet.",
            "Someone demonstrating yoga poses in a park.",
        ]
        captions_violation = [
            "The video contains nudity in violation of sexual content policy.",
            "The video depicts violent fighting which violates community guidelines.",
            "The content includes hate speech targeting a minority group.",
        ]
        for i in range(n):
            is_viol = i % 3 == 0
            viol_idx = i % len(self.VIOLATION_TYPES)
            labels = [0] * len(self.VIOLATION_TYPES)
            if is_viol:
                labels[viol_idx] = 1
            samples.append(
                {
                    "video_id": f"vid_{i:05d}",
                    "frames": None,  # will generate random tensors
                    "policy_text": policies[i % len(policies)],
                    "caption": (
                        captions_violation[i % len(captions_violation)]
                        if is_viol
                        else captions_safe[i % len(captions_safe)]
                    ),
                    "violation_labels": labels,
                    "is_violation": is_viol,
                }
            )
        return samples

    def _load_frames(
        self, frame_paths: Optional[List[str]]
    ) -> torch.Tensor:
        """Load video frames as (T, C, H, W) tensor."""
        if frame_paths is None or len(frame_paths) == 0:
            # Toy: random frames
            return torch.randn(self.num_frames, 3, 224, 224)

        frames = []
        indices = np.linspace(0, len(frame_paths) - 1, self.num_frames, dtype=int)
        for idx in indices:
            img = Image.open(frame_paths[idx]).convert("RGB")
            processed = self.image_processor(images=img, return_tensors="pt")
            frames.append(processed["pixel_values"].squeeze(0))
        return torch.stack(frames, dim=0)  # (T, C, H, W)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        # Video frames: (T, C, H, W)
        pixel_values = self._load_frames(sample.get("frames"))

        # Policy text encoding
        policy_enc = self.tokenizer(
            sample["policy_text"],
            max_length=self.max_policy_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Caption encoding (target)
        caption_enc = self.tokenizer(
            sample["caption"],
            max_length=self.max_caption_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = caption_enc["input_ids"].squeeze(0).clone()
        # Mask padding tokens in labels
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "video_id": sample["video_id"],
            "pixel_values": pixel_values,
            "policy_input_ids": policy_enc["input_ids"].squeeze(0),
            "caption_input_ids": caption_enc["input_ids"].squeeze(0),
            "labels": labels,
            "violation_labels": torch.tensor(
                sample["violation_labels"], dtype=torch.float
            ),
            "is_violation": torch.tensor(int(sample["is_violation"])),
        }


def build_dataloader(
    data_path: str,
    tokenizer,
    image_processor,
    split: str = "train",
    batch_size: int = 4,
    num_workers: int = 0,
) -> DataLoader:
    dataset = VideoModerationDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        image_processor=image_processor,
        split=split,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
    )
