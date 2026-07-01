from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MorePolicy(nn.Module):
    def __init__(self, vocab_size: int, hidden: int = 96, n_responses: int = 3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden, padding_idx=0)
        self.encoder = nn.GRU(hidden, hidden, batch_first=True)
        self.reasoning_head = nn.Linear(hidden, 2)
        self.response_head = nn.Linear(hidden, n_responses)
        self.log_reward_weights = nn.Parameter(torch.zeros(3))

    def forward(self, ids: torch.Tensor):
        x = self.emb(ids)
        _, h = self.encoder(x)
        h = h[-1]
        return self.reasoning_head(h), self.response_head(h)

    def adaptive_loss(self, reasoning_logits, response_logits, reasoning, response):
        reasoning_loss = F.cross_entropy(reasoning_logits, reasoning)
        response_loss = F.cross_entropy(response_logits, response)
        naturalness_proxy = -torch.softmax(response_logits, -1).amax(-1).mean()
        weights = torch.softmax(self.log_reward_weights, dim=0)
        return weights[0] * reasoning_loss + weights[1] * response_loss + weights[2] * naturalness_proxy


def metrics(reasoning_logits, response_logits, reasoning, response):
    reasoning_acc = (reasoning_logits.argmax(-1) == reasoning).float().mean().item()
    response_acc = (response_logits.argmax(-1) == response).float().mean().item()
    both = ((reasoning_logits.argmax(-1) == reasoning) & (response_logits.argmax(-1) == response)).float().mean().item()
    return reasoning_acc, response_acc, both
