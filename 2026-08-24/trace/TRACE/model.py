from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextEvidenceEncoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int = 96):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.projection = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.LayerNorm(hidden_size))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        mask = tokens.ne(0).float().unsqueeze(-1)
        pooled = (self.embedding(tokens) * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.projection(pooled)


class TRACEModel(nn.Module):
    def __init__(self, vocab_size: int, num_attributes: int, num_values: int, num_verdicts: int, hidden_size: int = 96):
        super().__init__()
        self.encoder = TextEvidenceEncoder(vocab_size, hidden_size)
        self.attribute_embedding = nn.Embedding(num_attributes, hidden_size)
        self.source_gate = nn.Sequential(nn.Linear(hidden_size * 4, hidden_size), nn.GELU(), nn.Linear(hidden_size, 4))
        self.value_head = nn.Linear(hidden_size * 2, num_values)
        self.verdict_head = nn.Linear(hidden_size * 2, num_verdicts)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        sku = self.encoder(batch["sku_tokens"])
        merchant = self.encoder(batch["merchant_tokens"])
        web = self.encoder(batch["web_tokens"])
        image = self.encoder(batch["image_tokens"])
        sources = torch.stack([sku, merchant, web, image], dim=1)
        source_logits = self.source_gate(torch.cat([sku, merchant, web, image], dim=-1))
        source_weights = F.softmax(source_logits, dim=-1).unsqueeze(-1)
        grounded_context = (sources * source_weights).sum(dim=1)
        attribute_context = self.attribute_embedding(batch["attribute"])
        joint = torch.cat([grounded_context, attribute_context], dim=-1)
        return {
            "value_logits": self.value_head(joint),
            "verdict_logits": self.verdict_head(joint),
            "source_weights": source_weights.squeeze(-1),
        }


def trace_loss(outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    value_loss = F.cross_entropy(outputs["value_logits"], batch["value"])
    verdict_loss = F.cross_entropy(outputs["verdict_logits"], batch["verdict"])
    entropy = -(outputs["source_weights"] * outputs["source_weights"].clamp_min(1e-6).log()).sum(dim=-1).mean()
    return value_loss + 0.7 * verdict_loss - 0.01 * entropy
