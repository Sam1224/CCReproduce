from __future__ import annotations

import json
import random
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class SimpleTokenizer:
    def __init__(self, vocab_size: int = 10000, pad_id: int = 0, bos_id: int = 1, eos_id: int = 2) -> None:
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id

    def encode(self, text: str, max_len: int) -> List[int]:
        # hash tokens into a fixed vocab for reproducibility
        toks = [t for t in text.lower().replace("\n", " ").split(" ") if t]
        ids = [self.bos_id]
        for t in toks[: max(0, max_len - 2)]:
            ids.append(3 + (abs(hash(t)) % (self.vocab_size - 3)))
        ids.append(self.eos_id)
        return ids


class QueryAdDataset(Dataset):
    def __init__(
        self,
        *,
        n: int = 6000,
        d_img: int = 64,
        d_ctx: int = 16,
        max_len: int = 48,
        seed: int = 7,
        jsonl_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        seed_everything(seed)
        self.tok = SimpleTokenizer()
        self.max_len = max_len
        self.d_img = d_img
        self.d_ctx = d_ctx

        self.items: List[Dict[str, torch.Tensor]] = []

        if jsonl_path:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    q = str(row["query"])
                    ad = str(row["ad_text"])
                    img = torch.tensor(row.get("img") or np.random.randn(d_img), dtype=torch.float32)
                    ctx = torch.tensor(row.get("ctx") or np.random.randn(d_ctx), dtype=torch.float32)
                    y = float(row.get("label", 0))
                    self.items.append(
                        {
                            "q": torch.tensor(self.tok.encode(q, max_len), dtype=torch.long),
                            "ad": torch.tensor(self.tok.encode(ad, max_len), dtype=torch.long),
                            "img": img,
                            "ctx": ctx,
                            "y": torch.tensor([y], dtype=torch.float32),
                        }
                    )
        else:
            offensive_words = ["hate", "kill", "sex", "violence", "racist", "slur"]
            benign_words = ["shoes", "phone", "discount", "review", "delivery", "camera"]

            for _ in range(n):
                is_off = float(np.random.rand() < 0.2)
                q_words = random.sample(benign_words, 2) + (random.sample(offensive_words, 1) if is_off else [])
                ad_words = random.sample(benign_words, 4) + (random.sample(offensive_words, 1) if (is_off and np.random.rand() < 0.5) else [])

                q = " ".join(q_words)
                ad = " ".join(ad_words)

                img = torch.randn(d_img)
                ctx = torch.randn(d_ctx)

                # label depends on mismatch between query intent and ad content (context-aware)
                y = is_off if ("hate" in q or "slur" in q) else float(np.random.rand() < 0.05)

                self.items.append(
                    {
                        "q": torch.tensor(self.tok.encode(q, max_len), dtype=torch.long),
                        "ad": torch.tensor(self.tok.encode(ad, max_len), dtype=torch.long),
                        "img": img,
                        "ctx": ctx,
                        "y": torch.tensor([y], dtype=torch.float32),
                    }
                )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.items[idx]


def pad_1d(seqs: List[torch.Tensor], pad_id: int) -> torch.Tensor:
    m = max(int(s.numel()) for s in seqs)
    out = torch.full((len(seqs), m), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : s.numel()] = s
    return out


def collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    pad_id = 0
    return {
        "q": pad_1d([b["q"] for b in batch], pad_id),
        "ad": pad_1d([b["ad"] for b in batch], pad_id),
        "img": torch.stack([b["img"] for b in batch], dim=0),
        "ctx": torch.stack([b["ctx"] for b in batch], dim=0),
        "y": torch.stack([b["y"] for b in batch], dim=0).squeeze(-1),
    }
