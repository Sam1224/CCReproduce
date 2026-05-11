"""
KuaiMod — SVP-style Dataset
Toy implementation of the Short Video Platform (SVP) Content Moderation benchmark.

Annotation schema (faithful to paper):
  - video_id: unique identifier
  - frames: list of PIL Image (or paths) representing sampled frames
  - text: associated text (caption / OCR / speech transcript)
  - label: {0: safe, 1: violating, 2: borderline}
  - cot_rationale: chain-of-thought reasoning string
  - verdict: final label (same as label but as string)
  - feedback_source: {user_report, reviewer, policy_update}
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset


LABEL2ID = {"safe": 0, "violating": 1, "borderline": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

POLICY_VIOLATIONS = [
    "violent content",
    "sexual content",
    "hate speech",
    "misleading health claims",
    "counterfeit product promotion",
    "gambling",
    "drug-related content",
]

SAMPLE_COT_TEMPLATES = {
    "safe": (
        "The content shows {context}. Reviewing against platform policies:\n"
        "1. Violence check: No violent acts or depictions found.\n"
        "2. Sexual content check: Content is appropriate for all ages.\n"
        "3. Harmful claims: No misleading health or financial claims detected.\n"
        "Verdict: SAFE. This content complies with all platform policies."
    ),
    "violating": (
        "The content shows {context}. Reviewing against platform policies:\n"
        "1. Policy match: Content appears to contain {violation}.\n"
        "2. Evidence: {evidence}\n"
        "3. Severity: High — direct violation of section {section} of platform rules.\n"
        "Verdict: VIOLATING. Action required: content removal."
    ),
    "borderline": (
        "The content shows {context}. Reviewing against platform policies:\n"
        "1. Ambiguous signals: {signal}\n"
        "2. Context consideration: {context_note}\n"
        "3. Policy grey area: This falls in the borderline region of {policy_area}.\n"
        "Verdict: BORDERLINE. Escalating for human reviewer assessment."
    ),
}


@dataclass
class SVPSample:
    video_id: str
    frames: List  # list of PIL Image or np arrays (toy: random tensors)
    text: str
    label: int  # 0=safe, 1=violating, 2=borderline
    cot_rationale: str
    verdict: str
    feedback_source: str = "reviewer"


def _generate_toy_frame(height: int = 224, width: int = 224) -> Image.Image:
    """Generate a random RGB image as a toy video frame."""
    data = torch.randint(0, 256, (height, width, 3), dtype=torch.uint8).numpy()
    return Image.fromarray(data, mode="RGB")


def _make_cot(label_str: str) -> str:
    template = SAMPLE_COT_TEMPLATES[label_str]
    if label_str == "safe":
        return template.format(context="a user demonstrating a cooking recipe")
    elif label_str == "violating":
        violation = random.choice(POLICY_VIOLATIONS)
        return template.format(
            context="a product advertisement",
            violation=violation,
            evidence="Text overlay contains unverified medical claims",
            section="4.2.1",
        )
    else:
        return template.format(
            context="an entertainment clip with edgy humor",
            signal="mild profanity in text; no visual violations",
            context_note="creator has clean historical record",
            policy_area="adult humor guidelines",
        )


class SVPDataset(Dataset):
    """
    Toy SVP dataset with the annotation schema from the KuaiMod paper.
    In production, frames come from real video segments sampled at 1 FPS.
    """

    def __init__(
        self,
        num_samples: int = 200,
        num_frames: int = 4,
        image_size: int = 224,
        split: str = "train",
        seed: int = 42,
    ):
        random.seed(seed)
        self.num_frames = num_frames
        self.image_size = image_size
        self.samples = self._generate(num_samples)

    def _generate(self, n: int) -> List[SVPSample]:
        label_strs = ["safe", "violating", "borderline"]
        weights = [0.5, 0.35, 0.15]  # realistic class imbalance
        samples = []
        for i in range(n):
            label_str = random.choices(label_strs, weights=weights)[0]
            sample = SVPSample(
                video_id=f"vid_{i:06d}",
                frames=[_generate_toy_frame(self.image_size, self.image_size)
                        for _ in range(self.num_frames)],
                text=f"Sample text for video {i}. " * random.randint(2, 8),
                label=LABEL2ID[label_str],
                cot_rationale=_make_cot(label_str),
                verdict=label_str,
                feedback_source=random.choice(["reviewer", "user_report", "policy_update"]),
            )
            samples.append(sample)
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> SVPSample:
        return self.samples[idx]


class SVPCollator:
    """
    Collates SVPSample batches for the KuaiMod model.
    In production, frames are processed by the VLM's image processor.
    Here we use simple resize + normalize as a toy.
    """

    def __init__(self, processor=None, max_text_len: int = 256):
        self.processor = processor
        self.max_text_len = max_text_len

    def __call__(self, batch: List[SVPSample]):
        import torchvision.transforms as T

        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        frame_tensors = []
        texts = []
        labels = []
        cot_rationales = []

        for sample in batch:
            # Stack frames: (num_frames, C, H, W)
            frames = torch.stack([transform(f) for f in sample.frames])
            frame_tensors.append(frames)
            texts.append(sample.text)
            labels.append(sample.label)
            cot_rationales.append(sample.cot_rationale)

        return {
            "frames": torch.stack(frame_tensors),  # (B, T, C, H, W)
            "texts": texts,
            "labels": torch.tensor(labels, dtype=torch.long),
            "cot_rationales": cot_rationales,
        }


if __name__ == "__main__":
    ds = SVPDataset(num_samples=50, split="train")
    print(f"Dataset size: {len(ds)}")
    sample = ds[0]
    print(f"video_id: {sample.video_id}")
    print(f"label: {ID2LABEL[sample.label]} ({sample.label})")
    print(f"cot_rationale:\n{sample.cot_rationale}")
    print(f"frames: {len(sample.frames)} frames, size={sample.frames[0].size}")
