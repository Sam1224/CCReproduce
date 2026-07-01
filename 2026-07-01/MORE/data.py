from __future__ import annotations

import random

import torch
from torch.utils.data import Dataset

ATTRS = ["credit_low", "credit_high", "eligible", "not_eligible", "vip", "new_user"]
RESPONSES = [
    "approved_natural",
    "rejected_natural",
    "ask_more_info",
]


class DialogVocab:
    def __init__(self):
        toks = ["[pad]", "[unk]", "installment", "refund", "coupon", "phone", "order", "can", "i", "buy", "with", "plan"] + ATTRS
        self.itos = toks
        self.stoi = {t: i for i, t in enumerate(toks)}

    def encode(self, text: str, max_len: int = 18):
        ids = [self.stoi.get(t, 1) for t in text.lower().replace("?", "").split()[:max_len]]
        ids += [0] * (max_len - len(ids))
        return torch.tensor(ids)


class MoreDialogDataset(Dataset):
    def __init__(self, n: int = 1200, seed: int = 0):
        rng = random.Random(seed)
        self.vocab = DialogVocab()
        self.samples = []
        for _ in range(n):
            credit_high = rng.random() > 0.45
            eligible = rng.random() > 0.25
            vip = rng.random() > 0.7
            attrs = ["credit_high" if credit_high else "credit_low", "eligible" if eligible else "not_eligible", "vip" if vip else "new_user"]
            question = rng.choice(["can i buy phone with installment plan", "can i refund order", "coupon for phone order"])
            reasoning_ok = int((credit_high and eligible) or vip)
            response = 0 if reasoning_ok else 1
            if "coupon" in question and not vip:
                response = 2
            text = question + " " + " ".join(attrs)
            self.samples.append({"ids": self.vocab.encode(text), "reasoning": reasoning_ok, "response": response})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate(batch):
    return {"ids": torch.stack([b["ids"] for b in batch]), "reasoning": torch.tensor([b["reasoning"] for b in batch]), "response": torch.tensor([b["response"] for b in batch])}
