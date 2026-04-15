import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHaystackRetriever(nn.Module):
    """A minimal dual-encoder retriever for MultiHaystack-like retrieval."""

    def __init__(self, vocab_size: int = 1000, d_model: int = 256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True),
            num_layers=2,
        )
        self.proj = nn.Linear(d_model, d_model)

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        h = self.encoder(x)
        pooled = h.mean(dim=1)
        z = F.normalize(self.proj(pooled), dim=-1)
        return z

    def forward(self, batch: dict) -> dict:
        q = self.encode(batch["query_ids"])  # [B, D]
        docs = self.encode(batch["doc_ids"])  # [N, D]
        scores = q @ docs.T  # [B, N]
        return {"scores": scores}


def info_nce_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """InfoNCE with a single positive per query."""
    return F.cross_entropy(scores, labels)
