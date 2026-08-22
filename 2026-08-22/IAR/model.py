from typing import Dict

import torch
from torch import nn


class TinyIARModel(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int = 96):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.decoder = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        _, hidden = self.encoder(embedded)
        return self.decoder(hidden.squeeze(0))


def merge_recover(domain_model: TinyIARModel, base_model: TinyIARModel, alpha: float = 0.65) -> Dict[str, torch.Tensor]:
    merged = {}
    for name, domain_weight in domain_model.state_dict().items():
        base_weight = base_model.state_dict()[name]
        merged[name] = alpha * domain_weight + (1.0 - alpha) * base_weight
    return merged
