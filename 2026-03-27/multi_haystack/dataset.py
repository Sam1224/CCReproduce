import random
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass
class MultiHaystackExample:
    query_ids: torch.Tensor
    doc_ids: torch.Tensor
    label: int


class ToyMultiHaystackDataset(Dataset):
    """Toy dataset for MultiHaystack: Benchmarking Multimodal Retrieval and Reasoning.

    The real MultiHaystack benchmark contains a large multimodal candidate pool
    (documents/images/videos) and open yet verifiable questions.

    This toy dataset mimics the *retrieval* core: each query has exactly one
    positive document in a small candidate set.
    """

    def __init__(self, num_queries: int = 128, num_docs: int = 256, seq_len: int = 32):
        self.num_queries = num_queries
        self.num_docs = num_docs
        self.seq_len = seq_len

        # Fixed doc token ids (like an indexed corpus)
        g = torch.Generator().manual_seed(42)
        self.doc_ids = torch.randint(0, 1000, (num_docs, seq_len), generator=g)

        # For each query, choose a positive doc id
        self.q_to_pos = [random.randrange(0, num_docs) for _ in range(num_queries)]

    def __len__(self) -> int:
        return self.num_queries

    def __getitem__(self, idx: int) -> MultiHaystackExample:
        pos = self.q_to_pos[idx]

        # Query ids are a noisy copy of the positive doc (simulates a relevant question)
        query_ids = self.doc_ids[pos].clone()
        noise = torch.randint(0, 1000, query_ids.shape)
        mask = torch.rand(query_ids.shape) < 0.15
        query_ids[mask] = noise[mask]

        return MultiHaystackExample(query_ids=query_ids, doc_ids=self.doc_ids, label=pos)


def collate_fn(batch: list[MultiHaystackExample]):
    return {
        "query_ids": torch.stack([b.query_ids for b in batch]),
        "doc_ids": batch[0].doc_ids,
        "label": torch.tensor([b.label for b in batch], dtype=torch.long),
    }
