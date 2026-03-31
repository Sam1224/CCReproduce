from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


SPECIAL = {
    "<pad>": 0,
    "<bos>": 1,
    "<eos>": 2,
    "<think>": 3,
    "</think>": 4,
    "<answer>": 5,
}


class SIDVocab:
    def __init__(self, n_items: int = 1000) -> None:
        self.n_items = n_items
        self.sid_offset = 16
        self.stoi = dict(SPECIAL)
        for i in range(6, self.sid_offset):
            self.stoi[f"<tok_{i}>"] = i
        self.itos = {i: s for s, i in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return self.sid_offset + self.n_items

    def sid_token_id(self, item_id: int) -> int:
        return self.sid_offset + int(item_id)

    def is_sid(self, token_id: int) -> bool:
        return token_id >= self.sid_offset


@dataclass
class SIDExample:
    prompt: torch.Tensor
    target_sid: int


class SyntheticSIDDataset(Dataset):
    """Synthetic generative recommendation with semantic IDs.

    Each sample contains:
      - a user history sequence of SIDs
      - a natural-language "reasoning" hint
      - requires the model to output the next-item SID
    """

    def __init__(
        self,
        *,
        n: int = 8000,
        n_items: int = 1000,
        max_hist: int = 20,
        seed: int = 7,
    ) -> None:
        super().__init__()
        seed_everything(seed)
        self.vocab = SIDVocab(n_items=n_items)

        self.items: List[SIDExample] = []
        for _ in range(n):
            hist_len = int(np.random.randint(5, max_hist + 1))
            hist = np.random.randint(0, n_items, size=(hist_len,)).tolist()
            target = (hist[-1] + int(np.random.randint(1, 10))) % n_items

            # Natural language hint is generated from a "semantic cluster".
            cluster = target % 8
            hint = [
                self.vocab.stoi["<bos>"],
                self.vocab.stoi["<think>"],
            ]
            # encode history as SID tokens
            hint.extend([self.vocab.sid_token_id(i) for i in hist])
            hint.extend(
                [
                    self.vocab.stoi["</think>"],
                    self.vocab.stoi["<tok_6>"] + cluster,
                    self.vocab.stoi["<answer>"],
                ]
            )

            self.items.append(SIDExample(prompt=torch.tensor(hint, dtype=torch.long), target_sid=target))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.items[idx]
        # For SFT, we train the model to generate: [prompt] <SID_target> <eos>
        y = torch.tensor([self.vocab.sid_token_id(ex.target_sid), self.vocab.stoi["<eos>"]], dtype=torch.long)
        ids = torch.cat([ex.prompt, y], dim=0)
        return {
            "ids": ids,
            "prompt_len": torch.tensor([int(ex.prompt.numel())], dtype=torch.long),
            "target_sid": torch.tensor([int(ex.target_sid)], dtype=torch.long),
        }


def pad_1d(seqs: List[torch.Tensor], pad: int) -> torch.Tensor:
    m = max(int(s.numel()) for s in seqs)
    out = torch.full((len(seqs), m), pad, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : s.numel()] = s
    return out


def collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    pad = SPECIAL["<pad>"]
    return {
        "ids": pad_1d([b["ids"] for b in batch], pad),
        "prompt_len": torch.cat([b["prompt_len"] for b in batch], dim=0),
        "target_sid": torch.cat([b["target_sid"] for b in batch], dim=0),
    }


def parse_prediction(seq: torch.Tensor, vocab: SIDVocab) -> Tuple[int, bool]:
    ids = seq.detach().cpu().tolist()
    try:
        ans = ids.index(SPECIAL["<answer>"])
    except ValueError:
        ans = -1

    sid = -1
    if ans >= 0:
        for t in ids[ans + 1 :]:
            if vocab.is_sid(t):
                sid = t - vocab.sid_offset
                break

    ok_format = ans >= 0 and sid >= 0
    return sid, ok_format
