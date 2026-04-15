from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from .teacher import Teacher
from .tokenizer import Tokenizer
from .utils import Batch, pad_2d


def build_prompt(a: int, b: int) -> list[str]:
    return ["ADD", str(a), "+", str(b), "="]


@dataclass(frozen=True)
class SFTExample:
    prompt: list[str]
    response: list[str]


class SFTDataset(Dataset[SFTExample]):
    def __init__(self, examples: list[SFTExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> SFTExample:
        return self.examples[idx]


def make_sft_dataset(
    tokenizer: Tokenizer,
    teacher: Teacher,
    n: int,
    seed: int,
) -> SFTDataset:
    g = torch.Generator().manual_seed(seed)
    examples: list[SFTExample] = []
    for _ in range(n):
        a = int(torch.randint(0, 10, (1,), generator=g).item())
        b = int(torch.randint(0, 10, (1,), generator=g).item())
        prompt = build_prompt(a, b)
        # teacher rollout: take argmax each step (deterministic).
        target = teacher._target_response_tokens(prompt)
        examples.append(SFTExample(prompt=prompt, response=target))
    return SFTDataset(examples)


def collate_sft(batch: list[SFTExample], tokenizer: Tokenizer) -> Batch:
    pad_id = tokenizer.token_to_id["<pad>"]
    bos_id = tokenizer.token_to_id["<bos>"]

    prompt_len = len(batch[0].prompt)

    seqs: list[list[int]] = []
    masks: list[list[bool]] = []

    for ex in batch:
        # input sequence: [prompt] + [<bos>] + response
        seq_tokens = ex.prompt + ["<bos>"] + ex.response
        seq_ids = tokenizer.encode(seq_tokens)
        seqs.append(seq_ids)

        # Only train on response tokens (including eos), i.e. positions after prompt.
        # We predict next token, so we mask on token positions that serve as targets.
        # Targets start at (prompt_len + 1)th token.
        loss_mask = [False] * (prompt_len + 1) + [True] * len(ex.response)
        masks.append(loss_mask)

    input_ids = pad_2d(seqs, pad_id)

    # pad mask to same length
    max_len = input_ids.shape[1]
    mask_tensor = torch.zeros((len(batch), max_len), dtype=torch.bool)
    for i, m in enumerate(masks):
        mask_tensor[i, : len(m)] = torch.tensor(m, dtype=torch.bool)

    return Batch(input_ids=input_ids, loss_mask=mask_tensor, prompt_len=prompt_len)


@dataclass(frozen=True)
class OPDExample:
    prompt: list[str]
    response: list[str]
    teacher_logp: torch.Tensor  # [T_resp]


class OPDDataset(Dataset[OPDExample]):
    def __init__(self, examples: list[OPDExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> OPDExample:
        return self.examples[idx]


def collate_opd(batch: list[OPDExample], tokenizer: Tokenizer) -> tuple[Batch, torch.Tensor]:
    pad_id = tokenizer.token_to_id["<pad>"]

    prompt_len = len(batch[0].prompt)

    seqs: list[list[int]] = []
    masks: list[list[bool]] = []
    teacher_logps: list[torch.Tensor] = []

    for ex in batch:
        seq_tokens = ex.prompt + ["<bos>"] + ex.response
        seqs.append(tokenizer.encode(seq_tokens))

        loss_mask = [False] * (prompt_len + 1) + [True] * len(ex.response)
        masks.append(loss_mask)
        teacher_logps.append(ex.teacher_logp)

    input_ids = pad_2d(seqs, pad_id)
    max_len = input_ids.shape[1]

    mask_tensor = torch.zeros((len(batch), max_len), dtype=torch.bool)
    for i, m in enumerate(masks):
        mask_tensor[i, : len(m)] = torch.tensor(m, dtype=torch.bool)

    # pad teacher logp to [B, T]
    teacher_pad = torch.full((len(batch), max_len), 0.0, dtype=torch.float32)
    for i, lp in enumerate(teacher_logps):
        start = prompt_len + 1
        teacher_pad[i, start : start + lp.shape[0]] = lp

    return Batch(input_ids=input_ids, loss_mask=mask_tensor, prompt_len=prompt_len), teacher_pad
