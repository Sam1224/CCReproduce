import math
import random
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class GenRecVocab:
    pad_id: int = 0
    bos_id: int = 1
    sep_id: int = 2
    eos_id: int = 3
    sid_base: int = 10


def encode_semantic_id(item_id: int, sid_len: int, vocab: GenRecVocab) -> list[int]:
    # Deterministic base-K encoding into a fixed-length token sequence.
    base = 97
    tokens: list[int] = []
    x = item_id
    for _ in range(sid_len):
        tokens.append(vocab.sid_base + (x % base))
        x //= base
    return tokens


class SyntheticGenRecDataset(Dataset):
    def __init__(
        self,
        num_users: int = 200,
        num_items: int = 1000,
        latent_dim: int = 32,
        history_len: int = 20,
        page_size: int = 10,
        sid_len: int = 3,
        num_samples: int = 4000,
        seed: int = 42,
    ):
        super().__init__()
        self.vocab = GenRecVocab()
        self.num_items = num_items
        self.history_len = history_len
        self.page_size = page_size
        self.sid_len = sid_len

        g = torch.Generator().manual_seed(seed)
        self.item_vecs = torch.randn(num_items, latent_dim, generator=g)
        self.user_vecs = torch.randn(num_users, latent_dim, generator=g)

        rng = random.Random(seed)
        self.samples = [rng.randrange(num_users) for _ in range(num_samples)]

    def __len__(self) -> int:
        return len(self.samples)

    def _sample_user_history(self, user_id: int) -> list[int]:
        # Sample history items from the user's preference distribution.
        logits = (self.item_vecs @ self.user_vecs[user_id]).float()
        probs = torch.softmax(logits, dim=0)
        items = torch.multinomial(probs, num_samples=self.history_len + self.page_size, replacement=True)
        return items.tolist()

    def __getitem__(self, idx: int) -> dict:
        user_id = self.samples[idx]
        items = self._sample_user_history(user_id)
        history_items = items[: self.history_len]
        target_items = items[self.history_len :]

        history_tokens: list[int] = [self.vocab.bos_id]
        for it in history_items:
            history_tokens.extend(encode_semantic_id(it, self.sid_len, self.vocab))
        history_tokens.append(self.vocab.sep_id)

        target_tokens: list[int] = []
        for it in target_items:
            target_tokens.extend(encode_semantic_id(it, self.sid_len, self.vocab))
        target_tokens.append(self.vocab.eos_id)

        input_ids = torch.tensor(history_tokens + target_tokens, dtype=torch.long)
        labels = torch.tensor([-100] * len(history_tokens) + target_tokens, dtype=torch.long)

        return {
            "user_id": user_id,
            "input_ids": input_ids,
            "labels": labels,
        }


def collate_batch(batch: list[dict]) -> dict:
    max_len = max(x["input_ids"].numel() for x in batch)
    vocab = GenRecVocab()

    input_ids = torch.full((len(batch), max_len), fill_value=vocab.pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), fill_value=-100, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)

    for i, ex in enumerate(batch):
        n = ex["input_ids"].numel()
        input_ids[i, :n] = ex["input_ids"]
        labels[i, :n] = ex["labels"]
        attention_mask[i, :n] = True

    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
