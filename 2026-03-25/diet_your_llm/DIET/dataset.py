from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Vocab:
    tokens: List[str]

    def __post_init__(self) -> None:  # pragma: no cover
        if len(set(self.tokens)) != len(self.tokens):
            raise ValueError("Duplicate tokens in vocab")

    @property
    def stoi(self) -> Dict[str, int]:
        return {t: i for i, t in enumerate(self.tokens)}

    @property
    def itos(self) -> Dict[int, str]:
        return {i: t for i, t in enumerate(self.tokens)}

    def encode(self, s: str) -> List[int]:
        out: List[int] = []
        for ch in s:
            if ch not in self.stoi:
                raise KeyError(f"Unknown token: {ch!r}")
            out.append(self.stoi[ch])
        return out

    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos[i] for i in ids)


def build_vocab() -> Vocab:
    # A tiny character-level vocabulary good enough for toy tasks.
    base = [
        "<pad>",
        "<bos>",
        "<eos>",
        " ",
        ":",
        "+",
        "=",
        "?",
        "A",  # add task tag
        "P",  # parity task tag
        "C",  # compare task tag
        "E",  # even
        "O",  # odd
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
    ]
    return Vocab(tokens=base)


def _clamp_0_9(x: int) -> int:
    return max(0, min(9, x))


class ToyTaskDataset(Dataset):
    """Three toy 'tasks' used to mimic task-specific activation profiling.

    Each sample is (prompt, answer_token). The model is trained to predict the
    single-character answer given the prompt.

    Tasks:
    - Add:  "A: 3+5=" -> "8"
    - Parity: "P: 4=" -> "E" / "O"
    - Compare: "C: 3+5?" -> "1" if a>b else "0"
    """

    def __init__(
        self,
        task: str,
        vocab: Vocab,
        num_samples: int = 512,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if task not in {"add", "parity", "compare"}:
            raise ValueError(f"Unknown task: {task}")
        self.task = task
        self.vocab = vocab
        self.num_samples = num_samples
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return self.num_samples

    def _sample_numbers(self) -> Tuple[int, int]:
        a = self.rng.randint(0, 9)
        b = self.rng.randint(0, 9)
        return a, b

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        del idx
        a, b = self._sample_numbers()

        if self.task == "add":
            prompt = f"A: {a}+{b}="
            ans = str(_clamp_0_9(a + b))
        elif self.task == "parity":
            prompt = f"P: {a}="
            ans = "E" if a % 2 == 0 else "O"
        else:
            prompt = f"C: {a}+{b}?"
            ans = "1" if a > b else "0"

        prompt_ids = self.vocab.encode(prompt)
        ans_id = self.vocab.stoi[ans]

        return {
            "task": self.task,
            "input_ids": torch.tensor(prompt_ids, dtype=torch.long),
            "target_id": torch.tensor(ans_id, dtype=torch.long),
        }


def collate_batch(samples: List[Dict[str, torch.Tensor]], pad_id: int) -> Dict[str, torch.Tensor]:
    max_len = max(int(s["input_ids"].numel()) for s in samples)
    input_ids = torch.full((len(samples), max_len), pad_id, dtype=torch.long)
    attn_mask = torch.zeros((len(samples), max_len), dtype=torch.bool)

    for i, s in enumerate(samples):
        ids = s["input_ids"]
        l = int(ids.numel())
        input_ids[i, :l] = ids
        attn_mask[i, :l] = True

    target_id = torch.stack([s["target_id"] for s in samples], dim=0)

    # 'task' is metadata; keep as python list.
    tasks = [str(s["task"]) for s in samples]

    return {
        "input_ids": input_ids,
        "attn_mask": attn_mask,
        "target_id": target_id,
        "task": tasks,
    }
