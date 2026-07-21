import random
from dataclasses import dataclass
from typing import Dict

import torch
from torch.utils.data import Dataset


STATE_NAMES = ["idle", "engage", "transform", "complete", "anomaly"]
ANOMALY_TYPES = ["none", "leakage", "breakage", "misalignment"]


@dataclass
class OvadExample:
    frames: torch.Tensor
    states: torch.Tensor
    frame_mask: torch.Tensor
    video_label: float
    anomaly_type: int
    anomaly_object: int


class OvadToyDataset(Dataset):
    def __init__(
        self,
        n: int = 1024,
        frames: int = 24,
        objects: int = 4,
        feat_dim: int = 8,
        seed: int = 13,
    ):
        self.rng = random.Random(seed)
        self.frames = frames
        self.objects = objects
        self.feat_dim = feat_dim
        self.samples = [self._make_one() for _ in range(n)]

    def _base_state(self, frame_idx: int) -> int:
        if frame_idx < 6:
            return 0
        if frame_idx < 12:
            return 1
        if frame_idx < 18:
            return 2
        return 3

    def _state_feature(self, state: int, object_idx: int) -> torch.Tensor:
        base = torch.zeros(self.feat_dim)
        base[state] = 1.0
        base[5] = object_idx / max(self.objects - 1, 1)
        base[6] = (state + object_idx) / 8.0
        base[7] = 1.0
        noise = torch.randn(self.feat_dim) * 0.03
        return base + noise

    def _anomaly_feature(self, anomaly_type: int, object_idx: int) -> torch.Tensor:
        feat = torch.zeros(self.feat_dim)
        feat[4] = 1.2
        feat[5] = object_idx / max(self.objects - 1, 1)
        feat[6] = 0.25 * anomaly_type
        feat[7] = 1.5
        if anomaly_type == 1:
            feat[1] = 0.8
        elif anomaly_type == 2:
            feat[2] = 0.8
        else:
            feat[3] = 0.8
        noise = torch.randn(self.feat_dim) * 0.04
        return feat + noise

    def _make_one(self) -> OvadExample:
        is_anomaly = self.rng.random() < 0.5
        anomaly_object = self.rng.randrange(self.objects) if is_anomaly else -1
        anomaly_type = self.rng.randrange(1, len(ANOMALY_TYPES)) if is_anomaly else 0
        onset = self.rng.randrange(14, self.frames - 3) if is_anomaly else self.frames

        clip = torch.zeros(self.frames, self.objects, self.feat_dim)
        states = torch.zeros(self.frames, self.objects, dtype=torch.long)
        frame_mask = torch.zeros(self.frames, 1)
        for time_idx in range(self.frames):
            for object_idx in range(self.objects):
                state = self._base_state(time_idx)
                feat = self._state_feature(state, object_idx)
                if is_anomaly and object_idx == anomaly_object and time_idx >= onset:
                    state = 4
                    feat = self._anomaly_feature(anomaly_type, object_idx)
                    frame_mask[time_idx, 0] = 1.0
                clip[time_idx, object_idx] = feat
                states[time_idx, object_idx] = state

        return OvadExample(
            frames=clip,
            states=states,
            frame_mask=frame_mask,
            video_label=1.0 if is_anomaly else 0.0,
            anomaly_type=anomaly_type,
            anomaly_object=anomaly_object,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        return {
            "frames": sample.frames,
            "states": sample.states,
            "frame_mask": sample.frame_mask,
            "video_label": torch.tensor([sample.video_label], dtype=torch.float32),
            "anomaly_type": torch.tensor(sample.anomaly_type, dtype=torch.long),
            "anomaly_object": torch.tensor(sample.anomaly_object, dtype=torch.long),
        }


def make_loaders(batch_size: int = 16):
    train = OvadToyDataset(n=768, seed=17)
    test = OvadToyDataset(n=192, seed=29)
    return (
        torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True),
        torch.utils.data.DataLoader(test, batch_size=batch_size),
    )
