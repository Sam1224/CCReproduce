from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import BOS, EOS, PAD


class SAMD2Q(nn.Module):
    def __init__(self, vocab_size: int = 256, image_dim: int = 64, hidden_dim: int = 128) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=PAD)
        self.title_encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.image_encoder = nn.Sequential(nn.Linear(image_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.fusion = nn.Linear(hidden_dim * 3, hidden_dim)
        self.decoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)
        self.value_head = nn.Linear(hidden_dim, 1)

    def encode(self, title_tokens: torch.Tensor, image_features: torch.Tensor) -> torch.Tensor:
        embedded = self.token_embedding(title_tokens)
        encoded, _ = self.title_encoder(embedded)
        mask = (title_tokens != PAD).unsqueeze(-1).float()
        text_repr = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        image_repr = self.image_encoder(image_features)
        return torch.tanh(self.fusion(torch.cat([text_repr, image_repr], dim=-1)))

    def forward(self, batch: Dict[str, torch.Tensor], use_masked_title: bool = False) -> torch.Tensor:
        title_key = "masked_title" if use_masked_title else "title"
        context = self.encode(batch[title_key], batch["image"])
        decoder_input = batch["target_query"][:, :-1]
        embedded = self.token_embedding(decoder_input)
        hidden0 = context.unsqueeze(0)
        decoded, _ = self.decoder(embedded, hidden0)
        return self.output(decoded)

    def generate(self, title_tokens: torch.Tensor, image_features: torch.Tensor, max_len: int = 8) -> torch.Tensor:
        context = self.encode(title_tokens, image_features)
        token = torch.full((title_tokens.size(0), 1), BOS, dtype=torch.long, device=title_tokens.device)
        hidden = context.unsqueeze(0)
        outputs = []
        for _ in range(max_len - 1):
            decoded, hidden = self.decoder(self.token_embedding(token[:, -1:]), hidden)
            next_token = self.output(decoded[:, -1]).argmax(dim=-1, keepdim=True)
            outputs.append(next_token)
            token = torch.cat([token, next_token], dim=1)
        return torch.cat(outputs, dim=1)


def sequence_loss(logits: torch.Tensor, target_query: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), target_query[:, 1:].reshape(-1), ignore_index=PAD)


def business_reward_loss(model: SAMD2Q, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
    logits = model(batch, use_masked_title=True)
    log_probs = F.log_softmax(logits, dim=-1)
    target = batch["target_query"][:, 1:]
    selected = torch.gather(log_probs, -1, target.unsqueeze(-1)).squeeze(-1)
    token_gain = (target != PAD).float().mean(dim=1)
    reward = 0.45 * batch["demand"] + 0.45 * batch["conversion"] + 0.10 * token_gain
    policy_loss = -(selected.mean(dim=1) * reward.detach()).mean()
    value = model.value_head(model.encode(batch["masked_title"], batch["image"])).squeeze(-1)
    value_loss = F.mse_loss(value, reward)
    return policy_loss + 0.1 * value_loss, {"reward": float(reward.mean().detach()), "policy_loss": float(policy_loss.detach())}
