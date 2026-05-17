"""
Toy e-commerce dataset for Valley3 training pipeline.
Interface-aligned with the full e-commerce dataset used in Valley3.
"""

import torch
from torch.utils.data import Dataset
from typing import Optional, Dict, Any
import random


# E-commerce task types (Valley3 §4 evaluation categories)
TASK_TYPES = [
    "product_understanding",   # Stage 3: attribute/category/title
    "livestream_analysis",     # Stage 3: host/product_demo/audience
    "moderation_governance",   # Stage 3: violation detection
    "consumer_insight",        # Stage 2/3: user intent
    "deep_research",           # Stage 4: agentic multi-turn
]

# Violation categories for e-commerce governance
VIOLATION_TYPES = [
    "false_efficacy_claim",    # 虚假功效声明
    "price_misleading",        # 价格误导
    "medical_claim",           # 医疗声明（违规）
    "counterfeit_indicator",   # 假冒商品指标
    "minor_inappropriate",     # 未成年人不适内容
    "compliant",               # 合规
]


class ToyEcomDataset(Dataset):
    """
    Toy dataset generating synthetic multimodal e-commerce samples.
    Each sample contains: text instruction, optional image, optional audio,
    and a target response appropriate for the task type.

    In production Valley3 training:
    - Stage 1: audio-text pairs from multilingual livestreams
    - Stage 2: interleaved image-text instruction data
    - Stage 3: e-commerce domain QA, violation annotation, product catalogs
    - Stage 4: multi-turn long-context e-commerce research dialogues
    """

    def __init__(
        self,
        stage: int = 1,
        num_samples: int = 100,
        seq_len: int = 512,
        image_size: int = 224,
        audio_frames: int = 3000,
        vocab_size: int = 151936,
        seed: int = 42,
    ):
        super().__init__()
        self.stage = stage
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.image_size = image_size
        self.audio_frames = audio_frames
        self.vocab_size = vocab_size
        random.seed(seed)
        torch.manual_seed(seed)

        self.samples = self._generate_samples()

    def _generate_samples(self):
        samples = []
        for i in range(self.num_samples):
            task = random.choice(TASK_TYPES)
            sample = {
                "task_type": task,
                "input_ids": torch.randint(0, self.vocab_size, (self.seq_len,)),
                "labels": torch.randint(0, self.vocab_size, (self.seq_len,)),
                "attention_mask": torch.ones(self.seq_len, dtype=torch.long),
            }

            if self.stage >= 1:
                # Stage 1: always include audio
                sample["mel_features"] = torch.randn(80, self.audio_frames)

            if self.stage >= 2:
                # Stage 2+: include image 80% of the time
                if random.random() < 0.8:
                    sample["pixel_values"] = torch.randn(3, self.image_size, self.image_size)

            if self.stage >= 3 and task == "moderation_governance":
                # Stage 3: violation label for governance tasks
                sample["violation_label"] = random.choice(range(len(VIOLATION_TYPES)))

            if self.stage >= 4:
                # Stage 4: long-context multi-turn (simulate with longer seq)
                sample["input_ids"] = torch.randint(0, self.vocab_size, (min(self.seq_len * 4, 2048),))
                sample["labels"] = torch.randint(0, self.vocab_size, (min(self.seq_len * 4, 2048),))
                sample["attention_mask"] = torch.ones(min(self.seq_len * 4, 2048), dtype=torch.long)

            samples.append(sample)
        return samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


def collate_fn(batch):
    """Pad and stack a batch of multimodal samples."""
    keys = batch[0].keys()
    result = {}
    for key in keys:
        if key == "task_type":
            result[key] = [s[key] for s in batch]
        elif isinstance(batch[0][key], torch.Tensor):
            if key in ("input_ids", "labels", "attention_mask"):
                # Pad to max length in batch
                max_len = max(s[key].size(0) for s in batch)
                padded = []
                for s in batch:
                    t = s[key]
                    pad_size = max_len - t.size(0)
                    if key == "labels":
                        padded.append(torch.cat([t, torch.full((pad_size,), -100)]))
                    elif key == "attention_mask":
                        padded.append(torch.cat([t, torch.zeros(pad_size, dtype=torch.long)]))
                    else:
                        padded.append(torch.cat([t, torch.zeros(pad_size, dtype=torch.long)]))
                result[key] = torch.stack(padded)
            elif key == "mel_features":
                # Pad audio to max T in batch
                max_t = max(s[key].size(1) for s in batch if key in s)
                padded = []
                for s in batch:
                    if key in s:
                        t = s[key]
                        pad_size = max_t - t.size(1)
                        padded.append(torch.cat([t, torch.zeros(80, pad_size)], dim=1))
                    else:
                        padded.append(torch.zeros(80, max_t))
                result[key] = torch.stack(padded)
            elif key == "pixel_values":
                # Batch images (some samples may not have images)
                imgs = [s.get(key, torch.zeros(3, 224, 224)) for s in batch]
                result[key] = torch.stack(imgs)
            elif key == "violation_label":
                labels = [s.get(key, -1) for s in batch]
                result[key] = torch.tensor(labels)
    return result
