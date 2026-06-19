"""Toy dataset for QueryAgent-R1 (interface-aligned with paper's data format)."""

import torch
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import List


@dataclass
class EComQuerySample:
    """One training sample: user behavior history → recommended query → product set."""
    item_ids: List[int]          # sequence of item IDs in user history
    behavior_types: List[int]    # 0=view, 1=click, 2=search, 3=purchase
    target_query: List[int]      # target query token IDs
    purchased_ids: List[int]     # items the user actually purchased


class EComQueryDataset(Dataset):
    """Toy e-commerce query recommendation dataset (interface-aligned)."""

    def __init__(
        self,
        num_users: int = 1000,
        max_history_len: int = 50,
        max_query_len: int = 16,
        max_purchases: int = 5,
        vocab_size: int = 32000,
        product_vocab: int = 10000,
        seed: int = 42,
    ):
        torch.manual_seed(seed)
        self.samples = []
        for _ in range(num_users):
            hist_len = torch.randint(10, max_history_len, (1,)).item()
            item_ids = torch.randint(1, product_vocab, (hist_len,)).tolist()
            behavior_types = torch.randint(0, 4, (hist_len,)).tolist()
            query_len = torch.randint(4, max_query_len, (1,)).item()
            # BOS=1, EOS=2, tokens in [3, vocab_size)
            target_query = [1] + torch.randint(3, vocab_size, (query_len,)).tolist() + [2]
            n_pur = torch.randint(1, max_purchases, (1,)).item()
            purchased_ids = torch.randint(1, product_vocab, (n_pur,)).tolist()
            self.samples.append(EComQuerySample(
                item_ids=item_ids,
                behavior_types=behavior_types,
                target_query=target_query,
                purchased_ids=purchased_ids,
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch: List[EComQuerySample], max_history: int = 50, max_query: int = 18):
    """Pad and collate a batch of EComQuerySamples."""
    # Pad item histories
    item_ids = torch.zeros(len(batch), max_history, dtype=torch.long)
    behavior_types = torch.zeros(len(batch), max_history, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_history, dtype=torch.float)

    max_q = max(len(s.target_query) for s in batch)
    max_q = min(max_q, max_query)
    target_queries = torch.zeros(len(batch), max_q, dtype=torch.long)

    max_p = max(len(s.purchased_ids) for s in batch)
    purchased = torch.zeros(len(batch), max_p, dtype=torch.long)

    for i, s in enumerate(batch):
        hist = s.item_ids[:max_history]
        item_ids[i, :len(hist)] = torch.tensor(hist)
        beh = s.behavior_types[:max_history]
        behavior_types[i, :len(beh)] = torch.tensor(beh)
        attention_mask[i, :len(hist)] = 1.0
        tq = s.target_query[:max_q]
        target_queries[i, :len(tq)] = torch.tensor(tq)
        pu = s.purchased_ids[:max_p]
        purchased[i, :len(pu)] = torch.tensor(pu)

    return {
        "item_ids": item_ids,
        "behavior_types": behavior_types,
        "attention_mask": attention_mask,
        "target_queries": target_queries,
        "purchased_ids": purchased,
    }


def get_dataloader(batch_size: int = 32, num_workers: int = 0) -> DataLoader:
    dataset = EComQueryDataset()
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(b), num_workers=num_workers
    )
