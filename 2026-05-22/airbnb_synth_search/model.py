import torch
import torch.nn as nn


class DualEncoder(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 128):
        super().__init__()
        self.emb = nn.EmbeddingBag(vocab_size, dim, mode="mean")
        self.proj = nn.Linear(dim, dim, bias=False)

    def encode(self, flat_tokens: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        x = self.emb(flat_tokens, offsets)
        x = self.proj(x)
        return nn.functional.normalize(x, dim=-1)

    def score(self, q: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        return (q * d).sum(dim=-1)


def pairwise_ranking_loss(pos_score: torch.Tensor, neg_score: torch.Tensor, margin: float = 0.2):
    target = torch.ones_like(pos_score)
    return nn.functional.margin_ranking_loss(pos_score, neg_score, target=target, margin=margin)
