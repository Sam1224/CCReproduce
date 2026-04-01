from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Sample:
    feats: torch.Tensor  # [F]
    target: torch.Tensor  # scalar float
    reasoning: str


class SimulaToyDataset(Dataset):
    """Toy dataset for reasoning-driven synthetic data generation.

    Paper: Reasoning-Driven Synthetic Data Generation and Evaluation (arXiv:2603.29791)

    We generate synthetic arithmetic word problems with a structured reasoning trace.
    """

    OPS = ["add", "sub", "mul"]

    def __init__(self, n: int = 20000, *, seed: int = 0, hard: bool = False) -> None:
        self.n = n
        rng = random.Random(seed)
        self._rows = []
        for _ in range(n):
            if hard:
                a = rng.randrange(50, 200)
                b = rng.randrange(50, 200)
            else:
                a = rng.randrange(0, 50)
                b = rng.randrange(0, 50)
            op = rng.choice(self.OPS)

            if op == "add":
                y = a + b
                trace = f"Compute sum: {a} + {b} = {y}."
            elif op == "sub":
                y = a - b
                trace = f"Compute difference: {a} - {b} = {y}."
            else:
                y = a * b
                trace = f"Compute product: {a} * {b} = {y}."

            op_onehot = [1.0 if op == k else 0.0 for k in self.OPS]
            feats = [a / 200.0, b / 200.0] + op_onehot
            self._rows.append((feats, float(y), trace))

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Sample:
        feats, y, trace = self._rows[idx]
        return Sample(
            feats=torch.tensor(feats, dtype=torch.float32),
            target=torch.tensor([y], dtype=torch.float32),
            reasoning=trace,
        )
