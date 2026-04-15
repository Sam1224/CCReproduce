import torch
import torch.nn as nn
import torch.nn.functional as F


class LLMJudgedRanker(nn.Module):
    """A tiny bi-encoder ranker trained on synthetic "LLM judgment" labels."""

    def __init__(self, vocab_size: int = 1000, d_model: int = 256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True),
            num_layers=2,
        )
        self.scorer = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def encode(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(ids)
        h = self.encoder(x)
        return h.mean(dim=1)

    def forward(self, batch: dict) -> torch.Tensor:
        q = self.encode(batch["query_ids"])
        d = self.encode(batch["item_ids"])
        logits = self.scorer(torch.cat([q, d], dim=-1))
        return logits


def bce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, labels)
