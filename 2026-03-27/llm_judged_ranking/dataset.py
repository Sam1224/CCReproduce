from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass
class RankingExample:
    query_ids: torch.Tensor
    item_ids: torch.Tensor
    label: torch.Tensor


class ToyLLMJudgedRankingDataset(Dataset):
    """Toy dataset for "Scaling Search Relevance ... with LLM-Generated Judgments".

    The real paper uses an in-domain LLM judge to generate millions of relevance
    labels. Here we mimic the setup with synthetic labels.
    """

    def __init__(self, num_samples: int = 2048, seq_len: int = 32):
        self.num_samples = num_samples
        self.seq_len = seq_len
        g = torch.Generator().manual_seed(7)

        self.query_ids = torch.randint(0, 1000, (num_samples, seq_len), generator=g)
        self.item_ids = torch.randint(0, 1000, (num_samples, seq_len), generator=g)

        # Synthetic label: higher overlap => more relevant
        overlap = (self.query_ids == self.item_ids).float().mean(dim=1)
        self.labels = (overlap > 0.03).float()

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> RankingExample:
        return RankingExample(
            query_ids=self.query_ids[idx],
            item_ids=self.item_ids[idx],
            label=self.labels[idx].unsqueeze(0),
        )


def collate_fn(batch: list[RankingExample]):
    return {
        "query_ids": torch.stack([b.query_ids for b in batch]),
        "item_ids": torch.stack([b.item_ids for b in batch]),
        "label": torch.stack([b.label for b in batch]),
    }
