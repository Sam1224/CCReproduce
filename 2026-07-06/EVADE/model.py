from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import CATEGORIES, RULES


class RuleAwareFusion(nn.Module):
    def __init__(self, hidden_size: int, num_rules: int):
        super().__init__()
        self.rule_embeddings = nn.Parameter(torch.randn(num_rules, hidden_size) * 0.02)
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

    def forward(self, content_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.query(content_state).unsqueeze(1)
        keys = self.key(self.rule_embeddings).unsqueeze(0)
        values = self.value(self.rule_embeddings).unsqueeze(0)
        attention = torch.softmax((query * keys).sum(dim=-1) / content_state.size(-1) ** 0.5, dim=-1)
        rule_context = torch.bmm(attention.unsqueeze(1), values.expand(content_state.size(0), -1, -1)).squeeze(1)
        return content_state + rule_context, attention


class EvadeModerator(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int = 96, pad_id: int = 0, image_dim: int = 4):
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_id)
        self.text_encoder = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.text_proj = nn.Linear(hidden_size * 2, hidden_size)
        self.image_proj = nn.Sequential(nn.Linear(image_dim, hidden_size), nn.ReLU(), nn.LayerNorm(hidden_size))
        self.fusion = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.ReLU(), nn.Dropout(0.1))
        self.rule_attention = RuleAwareFusion(hidden_size, len(RULES))
        self.rule_head = nn.Linear(hidden_size, len(RULES))
        self.category_head = nn.Linear(hidden_size, len(CATEGORIES))

    def encode_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        mask = (input_ids != self.pad_id).float().unsqueeze(-1)
        embedded = self.embedding(input_ids)
        encoded, _ = self.text_encoder(embedded)
        pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return torch.relu(self.text_proj(pooled))

    def forward(self, input_ids: torch.Tensor, image_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        text_state = self.encode_text(input_ids)
        image_state = self.image_proj(image_features)
        fused = self.fusion(torch.cat([text_state, image_state], dim=-1))
        rule_enhanced, attention = self.rule_attention(fused)
        return {
            "category_logits": self.category_head(rule_enhanced),
            "rule_logits": self.rule_head(rule_enhanced),
            "rule_attention": attention,
        }

    def loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self(batch["input_ids"], batch["image_features"])
        category_loss = F.cross_entropy(outputs["category_logits"], batch["label"])
        rule_loss = F.binary_cross_entropy_with_logits(outputs["rule_logits"], batch["rule_targets"])
        return category_loss + 0.6 * rule_loss

    def explain(self, input_ids: torch.Tensor, image_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self(input_ids, image_features)
        return {
            "category": outputs["category_logits"].argmax(dim=-1),
            "rules": (torch.sigmoid(outputs["rule_logits"]) > 0.5).long(),
            "rule_attention": outputs["rule_attention"],
        }
