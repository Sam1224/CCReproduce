from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ToyModerationConfig:
    vocab_size: int = 256
    seq_len: int = 64
    trigger_tokens: tuple[int, ...] = (13, 37, 101)
    p_trigger: float = 0.35


class ToyModerationDataset(Dataset):
    """Synthetic dataset for streaming moderation.

    Each sample is a token sequence.

    - A sequence is considered unsafe if it contains any `trigger_tokens`.
    - For streaming labels, y[t]=1 iff token t is a trigger token.

    This matches the 'moderate during generation' setup: the model should flag
    as soon as the unsafe signal appears.
    """

    def __init__(self, n: int, cfg: ToyModerationConfig, seed: int = 0):
        self.n = n
        self.cfg = cfg
        g = torch.Generator().manual_seed(seed)

        x = torch.randint(0, cfg.vocab_size, (n, cfg.seq_len), generator=g)

        # inject triggers into a subset of sequences
        mask = torch.rand(n, generator=g) < cfg.p_trigger
        idx = torch.where(mask)[0]
        if len(idx) > 0:
            pos = torch.randint(0, cfg.seq_len, (len(idx),), generator=g)
            trig = torch.tensor(cfg.trigger_tokens, dtype=torch.long)
            trig_pick = trig[torch.randint(0, len(trig), (len(idx),), generator=g)]
            x[idx, pos] = trig_pick

        # streaming labels: prefix-has-trigger
        is_trigger = torch.zeros_like(x, dtype=torch.bool)
        for t in cfg.trigger_tokens:
            is_trigger |= x.eq(t)
        # token-level label: 1 only at the first appearance of an unsafe trigger token
        # (streaming moderation aims to raise an alert as soon as it appears).
        y_stream = is_trigger.to(torch.float32)

        self.x = x
        self.y_stream = y_stream

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        return {
            "x": self.x[idx],
            "y_stream": self.y_stream[idx],
        }
