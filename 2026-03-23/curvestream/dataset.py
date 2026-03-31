from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch.utils.data import Dataset


@dataclass
class CurveStreamExample:
    frames: torch.Tensor  # [T, D]
    label: int
    event_index: int


@dataclass
class CurveStreamBatch:
    frames: torch.Tensor  # [B, T, D]
    labels: torch.Tensor  # [B]
    event_index: torch.Tensor  # [B]


def _unit(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / (v.norm(dim=-1, keepdim=True) + eps)


def _make_class_directions(num_classes: int, dim: int, *, device: torch.device) -> torch.Tensor:
    g = torch.Generator(device=device)
    g.manual_seed(20260323)
    dirs = torch.randn(num_classes, dim, generator=g, device=device)
    return _unit(dirs)


def generate_sequence(
    *,
    label: int,
    num_classes: int,
    seq_len: int,
    dim: int,
    noise_std: float,
    generator: torch.Generator,
    device: torch.device,
) -> CurveStreamExample:
    """Generate a toy streaming-video feature trajectory.

    Intuition (toy approximation of CurveStream):
    - The sequence contains an "event" where the latent direction changes.
    - This creates high curvature in the feature trajectory.
    - A curvature-aware memory policy should preserve frames around the event.

    This is NOT a real video model; it is only a controllable proxy task.
    """

    class_dirs = _make_class_directions(num_classes, dim, device=device)

    # Pick a turning point (event) not too close to the edges.
    event_index = int(torch.randint(low=6, high=max(7, seq_len - 6), size=(1,), generator=generator, device=device).item())

    # Two directions: before-event (random) and after-event (label-conditioned).
    d0 = _unit(torch.randn(dim, generator=generator, device=device))
    d1 = class_dirs[label]

    # Build a smooth velocity curve that turns around event_index.
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    sharpness = 1.2
    gate = torch.sigmoid((t - float(event_index)) / sharpness).unsqueeze(-1)  # [T,1]
    vel = _unit((1 - gate) * d0 + gate * d1)  # [T,D]

    # Add a small periodic component to create mild curvature elsewhere.
    phase = torch.rand(1, generator=generator, device=device) * 2 * math.pi
    wobble = 0.15 * torch.sin(t / 3.5 + phase).unsqueeze(-1) * _unit(torch.randn(dim, generator=generator, device=device))
    vel = _unit(vel + wobble)

    # Integrate velocity to get a position-like feature sequence.
    frames = torch.cumsum(vel, dim=0)
    frames = frames + noise_std * torch.randn_like(frames, generator=generator)

    return CurveStreamExample(frames=frames, label=int(label), event_index=event_index)


class CurveStreamToyDataset(Dataset[CurveStreamExample]):
    def __init__(
        self,
        *,
        n_samples: int,
        num_classes: int = 6,
        seq_len: int = 32,
        dim: int = 32,
        noise_std: float = 0.08,
        seed: int = 0,
        device: str | torch.device = "cpu",
    ) -> None:
        self.n_samples = int(n_samples)
        self.num_classes = int(num_classes)
        self.seq_len = int(seq_len)
        self.dim = int(dim)
        self.noise_std = float(noise_std)
        self.seed = int(seed)
        self.device = torch.device(device)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> CurveStreamExample:
        g = torch.Generator(device=self.device)
        g.manual_seed(self.seed * 10_000 + int(idx))
        label = int(torch.randint(low=0, high=self.num_classes, size=(1,), generator=g, device=self.device).item())
        return generate_sequence(
            label=label,
            num_classes=self.num_classes,
            seq_len=self.seq_len,
            dim=self.dim,
            noise_std=self.noise_std,
            generator=g,
            device=self.device,
        )


def collate(examples: List[CurveStreamExample]) -> CurveStreamBatch:
    frames = torch.stack([ex.frames for ex in examples], dim=0)
    labels = torch.tensor([ex.label for ex in examples], dtype=torch.long, device=frames.device)
    event_index = torch.tensor([ex.event_index for ex in examples], dtype=torch.long, device=frames.device)
    return CurveStreamBatch(frames=frames, labels=labels, event_index=event_index)


def split_dataset(
    ds: CurveStreamToyDataset, *, frac: float = 0.8
) -> Tuple[CurveStreamToyDataset, CurveStreamToyDataset]:
    n_train = int(round(len(ds) * float(frac)))
    n_train = max(1, min(len(ds) - 1, n_train))
    train = CurveStreamToyDataset(
        n_samples=n_train,
        num_classes=ds.num_classes,
        seq_len=ds.seq_len,
        dim=ds.dim,
        noise_std=ds.noise_std,
        seed=ds.seed,
        device=ds.device,
    )
    test = CurveStreamToyDataset(
        n_samples=len(ds) - n_train,
        num_classes=ds.num_classes,
        seq_len=ds.seq_len,
        dim=ds.dim,
        noise_std=ds.noise_std,
        seed=ds.seed + 999,
        device=ds.device,
    )
    return train, test
