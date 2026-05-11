"""
Toy dataset for Valley3 four-stage training.
Interface-aligned with the paper's data format for each stage.

Stage 1 data: audio-text pairs (speech recognition, audio QA)
Stage 2 data: multimodal instruction-following (image/video/audio + instruction)
Stage 3 data: e-commerce domain (product QA, compliance check, short-video analysis)
Stage 4 data: long-context e-commerce reasoning (multi-turn, tool-use)
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple
import random


class Stage1AudioDataset(Dataset):
    """
    Stage 1: Audio understanding pre-training.
    Pairs: (audio mel spectrogram, transcription text)
    E-commerce focus: short-video narrations, product descriptions read aloud.
    """

    def __init__(self, size: int = 100, vocab_size: int = 32000):
        self.size = size
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Toy mel spectrogram: [80, 3000] → shorter for toy
        mel_spectrogram = torch.randn(80, 300)
        # Toy text tokens (audio transcription)
        seq_len = random.randint(8, 32)
        input_ids = torch.randint(0, self.vocab_size, (seq_len,))
        labels = input_ids.clone()
        labels[:3] = -100  # Ignore audio prompt tokens in loss

        return {
            "mel_spectrograms": mel_spectrogram,
            "input_ids": input_ids,
            "labels": labels,
        }


class Stage2CrossModalDataset(Dataset):
    """
    Stage 2: Cross-modal instruction following.
    Format: (image/video/audio, instruction) → response
    Covers: product image understanding, video captioning, audio QA.
    """

    INSTRUCTIONS = [
        "这个商品图片的主要特点是什么？",
        "视频中的商品是否符合平台规范？",
        "请描述这段音频中讲述的商品信息。",
        "这张图片中的品牌是否清晰可见？",
        "请总结这段短视频的核心卖点。",
    ]

    def __init__(self, size: int = 100, vocab_size: int = 32000):
        self.size = size
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        modality = random.choice(["image", "video", "audio"])

        sample = {}

        if modality in ("image", "video"):
            num_frames = 1 if modality == "image" else random.randint(2, 8)
            # Tiny pixel values: [num_frames, 3, 32, 32] for toy
            sample["pixel_values"] = torch.randn(num_frames, 3, 32, 32)

        if modality == "audio":
            sample["mel_spectrograms"] = torch.randn(80, 300)

        seq_len = random.randint(16, 48)
        sample["input_ids"] = torch.randint(0, self.vocab_size, (seq_len,))
        sample["labels"] = sample["input_ids"].clone()
        sample["labels"][:8] = -100  # Ignore instruction tokens in loss

        return sample


class Stage3EcommerceDataset(Dataset):
    """
    Stage 3: E-commerce domain knowledge injection.
    Tasks:
      - Product attribute extraction
      - Compliance / violation detection (虚假宣传、违禁品)
      - Cross-border product description generation
      - Short-video e-commerce content understanding
      - Influencer (达人) content quality assessment
    """

    ECOM_TASKS = [
        "product_attribute_extraction",
        "violation_detection",
        "description_generation",
        "short_video_understanding",
        "influencer_quality_assessment",
    ]

    def __init__(self, size: int = 200, vocab_size: int = 32000):
        self.size = size
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        task = random.choice(self.ECOM_TASKS)

        sample = {"task": task}

        # All e-commerce tasks involve images (product images or video frames)
        num_frames = random.randint(1, 4)
        sample["pixel_values"] = torch.randn(num_frames, 3, 32, 32)

        # Short-video tasks also include audio
        if task in ("short_video_understanding", "influencer_quality_assessment"):
            sample["mel_spectrograms"] = torch.randn(80, 300)

        seq_len = random.randint(32, 64)
        sample["input_ids"] = torch.randint(0, self.vocab_size, (seq_len,))
        sample["labels"] = sample["input_ids"].clone()
        sample["labels"][:12] = -100

        return sample


class Stage4LongContextDataset(Dataset):
    """
    Stage 4: Long-context e-commerce reasoning.
    Tasks: multi-turn product research, competitive analysis, deep compliance review.
    Simulates agentic search trajectories (tool call + response sequences).
    """

    def __init__(self, size: int = 50, vocab_size: int = 32000, max_len: int = 512):
        self.size = size
        self.vocab_size = vocab_size
        self.max_len = max_len

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Long sequences with multiple image segments
        num_product_images = random.randint(2, 5)
        sample = {
            "pixel_values": torch.randn(num_product_images, 3, 32, 32),
        }

        # Long conversation (agentic search trajectory)
        seq_len = random.randint(128, self.max_len)
        sample["input_ids"] = torch.randint(0, self.vocab_size, (seq_len,))
        sample["labels"] = sample["input_ids"].clone()
        # Ignore system prompt + tool call tokens
        sample["labels"][:32] = -100

        return sample


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Pad and batch samples from any stage."""
    keys = batch[0].keys()
    result = {}

    for key in keys:
        if key == "task":
            continue
        tensors = [item[key] for item in batch if key in item]
        if not tensors:
            continue

        if key == "pixel_values":
            # Pad to max num_frames in batch (first dim after batch)
            max_frames = max(t.shape[0] for t in tensors)
            padded = []
            for t in tensors:
                if t.shape[0] < max_frames:
                    pad = torch.zeros(max_frames - t.shape[0], *t.shape[1:])
                    t = torch.cat([t, pad], dim=0)
                padded.append(t)
            result[key] = torch.stack(padded)
        elif key in ("input_ids", "labels"):
            # Pad sequences to max length
            max_len = max(t.shape[0] for t in tensors)
            padded = []
            for t in tensors:
                if t.shape[0] < max_len:
                    pad_val = -100 if key == "labels" else 0
                    pad = torch.full((max_len - t.shape[0],), pad_val, dtype=t.dtype)
                    t = torch.cat([t, pad])
                padded.append(t)
            result[key] = torch.stack(padded)
        elif key == "mel_spectrograms":
            result[key] = torch.stack(tensors)

    return result


def get_dataloader(stage: int, batch_size: int = 4, size: int = 100) -> DataLoader:
    """Get dataloader for a specific training stage."""
    datasets = {
        1: Stage1AudioDataset(size=size),
        2: Stage2CrossModalDataset(size=size),
        3: Stage3EcommerceDataset(size=size),
        4: Stage4LongContextDataset(size=size),
    }
    assert stage in datasets, f"Stage must be 1-4, got {stage}"
    return DataLoader(
        datasets[stage],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
