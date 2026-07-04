from typing import List

import torch
from torch import nn

from data import CATEGORIES


class GrocLM(nn.Module):
    def __init__(self, vocab_size: int, category_count: int, hidden_size: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.history_encoder = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.query_projection = nn.Linear(hidden_size, hidden_size)
        self.rebuy_gate = nn.Sequential(nn.Linear(category_count, hidden_size), nn.ReLU(), nn.Linear(hidden_size, category_count))
        self.classifier = nn.Linear(hidden_size, category_count)

    def forward(self, history: torch.Tensor, query: torch.Tensor, rebuy_prior: torch.Tensor) -> torch.Tensor:
        history_vectors = self.embedding(history)
        _, hidden = self.history_encoder(history_vectors)
        query_vector = self.query_projection(self.embedding(query))
        fused = torch.tanh(hidden.squeeze(0) + query_vector)
        relation_logits = self.classifier(fused)
        rebuy_logits = self.rebuy_gate(rebuy_prior)
        return relation_logits + rebuy_logits

    def constrained_decode(self, logits: torch.Tensor, top_k: int = 5) -> List[List[str]]:
        probabilities = torch.sigmoid(logits)
        selected = torch.topk(probabilities, k=min(top_k, len(CATEGORIES)), dim=-1).indices
        return [[CATEGORIES[index] for index in row.tolist()] for row in selected]
