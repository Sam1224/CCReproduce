from dataclasses import dataclass
from typing import Dict

import torch
from torch.utils.data import Dataset


CLASS_NAMES = ["safe", "copyright", "gambling", "adult", "scam"]


@dataclass
class ReferenceBank:
    frames: torch.Tensor
    audio: torch.Tensor
    text: torch.Tensor
    labels: torch.Tensor


class ToyLivestreamModerationDataset(Dataset):
    def __init__(
        self,
        length: int = 512,
        num_frames: int = 6,
        visual_dim: int = 32,
        audio_dim: int = 16,
        text_dim: int = 24,
        num_references_per_class: int = 6,
        seed: int = 7,
    ):
        self.length = length
        self.num_frames = num_frames
        self.visual_dim = visual_dim
        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.num_classes = len(CLASS_NAMES)
        self.num_references_per_class = num_references_per_class
        generator = torch.Generator().manual_seed(seed)
        self.visual_prototypes = torch.randn(self.num_classes, num_frames, visual_dim, generator=generator)
        self.audio_prototypes = torch.randn(self.num_classes, audio_dim, generator=generator)
        self.text_prototypes = torch.randn(self.num_classes, text_dim, generator=generator)
        self.reference_bank = self._build_reference_bank()

    def _make_clip(self, label: int, variant: int) -> Dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(label * 10_000 + variant)
        frames = self.visual_prototypes[label] + 0.18 * torch.randn(self.num_frames, self.visual_dim, generator=generator)
        audio = self.audio_prototypes[label] + 0.16 * torch.randn(self.audio_dim, generator=generator)
        text = self.text_prototypes[label] + 0.12 * torch.randn(self.text_dim, generator=generator)
        if label != 0:
            drift_label = (label % (self.num_classes - 1)) + 1
            frames = frames + 0.06 * self.visual_prototypes[drift_label]
            audio = audio + 0.04 * self.audio_prototypes[drift_label]
            text = text + 0.05 * self.text_prototypes[drift_label]
        fused_target = torch.cat([frames.mean(dim=0), audio, text, frames.std(dim=0)[:24]], dim=0)
        logits = []
        for class_id in range(self.num_classes):
            prototype = torch.cat(
                [
                    self.visual_prototypes[class_id].mean(dim=0),
                    self.audio_prototypes[class_id],
                    self.text_prototypes[class_id],
                    self.visual_prototypes[class_id].std(dim=0)[:24],
                ],
                dim=0,
            )
            logits.append(torch.cosine_similarity(fused_target, prototype, dim=0))
        teacher_logits = torch.stack(logits) * 5.0
        return {
            "frames": frames.float(),
            "audio": audio.float(),
            "text": text.float(),
            "label": torch.tensor(label, dtype=torch.long),
            "teacher_hidden": fused_target.float(),
            "teacher_logits": teacher_logits.float(),
            "risk_target": torch.tensor(0 if label == 0 else 1, dtype=torch.long),
        }

    def _build_reference_bank(self) -> ReferenceBank:
        frames = []
        audio = []
        text = []
        labels = []
        for label in range(1, self.num_classes):
            for variant in range(self.num_references_per_class):
                sample = self._make_clip(label, 50_000 + label * 100 + variant)
                frames.append(sample["frames"])
                audio.append(sample["audio"])
                text.append(sample["text"])
                labels.append(sample["label"])
        return ReferenceBank(
            frames=torch.stack(frames),
            audio=torch.stack(audio),
            text=torch.stack(text),
            labels=torch.stack(labels),
        )

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        label = index % self.num_classes
        return self._make_clip(label=label, variant=index)

    def build_reference_bank(self) -> ReferenceBank:
        return self.reference_bank
