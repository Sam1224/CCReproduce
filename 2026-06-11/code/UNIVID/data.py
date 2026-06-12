"""
UNIVID Data Pipeline
Toy dataset interface aligned with the paper's training setup.
"""

import os
import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import List, Dict, Optional, Tuple
import numpy as np


VIOLATION_LABELS = [
    "violence", "nudity", "hate_speech", "spam",
    "misinformation", "drug", "gambling", "copyright",
    "safe",  # non-violative
]


class VideoModerationDataset(Dataset):
    """
    Dataset for UNIVID caption generation and moderation training.
    
    Each sample:
        - frames: List[PIL.Image] (sampled at 1fps, max 16 frames)
        - policy_prompt: str  (platform policy description)
        - caption: str        (expert-annotated policy-aware caption)
        - label: int          (violation class index, or safe)
        - is_violative: bool
    
    For production: replace with real video loading + expert annotations.
    Toy version generates synthetic data for interface validation.
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        num_frames: int = 8,
        frame_size: Tuple[int, int] = (224, 224),
        toy_size: int = 200,
    ):
        self.num_frames = num_frames
        self.frame_size = frame_size

        if not os.path.exists(data_root):
            # Generate toy data
            self.samples = self._generate_toy_samples(toy_size)
        else:
            meta_path = os.path.join(data_root, f"{split}_meta.json")
            with open(meta_path) as f:
                self.samples = json.load(f)

    def _generate_toy_samples(self, n: int) -> List[Dict]:
        """Generate synthetic samples for interface testing."""
        samples = []
        for i in range(n):
            label_idx = random.randint(0, len(VIOLATION_LABELS) - 1)
            label = VIOLATION_LABELS[label_idx]
            is_violative = label != "safe"

            samples.append({
                "video_id": f"toy_{i:05d}",
                "frames": None,  # generated on-the-fly
                "policy_prompt": "Platform policy: no violence, nudity, hate speech, misinformation, or illegal content.",
                "caption": self._toy_caption(label),
                "label": label_idx,
                "label_name": label,
                "is_violative": is_violative,
            })
        return samples

    def _toy_caption(self, label: str) -> str:
        templates = {
            "violence": "The video shows physical altercation between individuals with visible aggressive behavior.",
            "nudity": "The video contains explicit content showing undressed individuals.",
            "hate_speech": "The video includes verbal statements targeting individuals based on ethnicity.",
            "spam": "The video is a repetitive commercial advertisement with misleading claims.",
            "misinformation": "The video presents unverified health claims without scientific basis.",
            "drug": "The video depicts individuals using controlled substances.",
            "gambling": "The video promotes online gambling services to underage audiences.",
            "copyright": "The video plays copyrighted music without authorization.",
            "safe": "The video shows ordinary daily activities with no policy violations observed.",
        }
        return templates.get(label, "Content description unavailable.")

    def _generate_toy_frame(self) -> Image.Image:
        """Generate a random RGB image as a toy video frame."""
        arr = np.random.randint(0, 256, (*self.frame_size, 3), dtype=np.uint8)
        return Image.fromarray(arr)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        # Load or generate frames
        if sample["frames"] is None:
            frames = [self._generate_toy_frame() for _ in range(self.num_frames)]
        else:
            frames = self._load_video_frames(sample["frames"])

        return {
            "video_id": sample["video_id"],
            "frames": frames,
            "policy_prompt": sample["policy_prompt"],
            "caption": sample["caption"],
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "is_violative": torch.tensor(sample["is_violative"], dtype=torch.float),
        }

    def _load_video_frames(self, frame_paths: List[str]) -> List[Image.Image]:
        frames = []
        indices = np.linspace(0, len(frame_paths) - 1, self.num_frames, dtype=int)
        for i in indices:
            img = Image.open(frame_paths[i]).convert("RGB").resize(self.frame_size)
            frames.append(img)
        return frames


def get_dataloader(
    data_root: str,
    split: str = "train",
    batch_size: int = 8,
    num_workers: int = 4,
    **kwargs,
) -> DataLoader:
    dataset = VideoModerationDataset(data_root, split, **kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        collate_fn=_collate_fn,
    )


def _collate_fn(batch: List[Dict]) -> Dict:
    """Custom collation: frames are lists of PIL Images, not tensors."""
    return {
        "video_ids": [s["video_id"] for s in batch],
        "frames": [s["frames"] for s in batch],  # list of lists
        "policy_prompts": [s["policy_prompt"] for s in batch],
        "captions": [s["caption"] for s in batch],
        "labels": torch.stack([s["label"] for s in batch]),
        "is_violative": torch.stack([s["is_violative"] for s in batch]),
    }


if __name__ == "__main__":
    ds = VideoModerationDataset("./toy_data", toy_size=50)
    print(f"Dataset size: {len(ds)}")
    sample = ds[0]
    print(f"Sample: video_id={sample['video_id']}, label={sample['label']}")
    print(f"Caption: {sample['caption'][:80]}...")
    print(f"Frames: {len(sample['frames'])} frames of size {sample['frames'][0].size}")
