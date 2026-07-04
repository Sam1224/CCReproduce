import torch
from torch import nn

from data import ATTRIBUTES


class EcomVLMAdapter(nn.Module):
    def __init__(self, vocab_size: int, image_feature_size: int, hidden_size: int = 64):
        super().__init__()
        self.text_embedding = nn.Embedding(vocab_size, hidden_size)
        self.image_adapter = nn.Sequential(nn.Linear(image_feature_size, hidden_size), nn.GELU(), nn.Linear(hidden_size, hidden_size))
        self.fusion = nn.Sequential(nn.LayerNorm(hidden_size * 2), nn.Linear(hidden_size * 2, hidden_size), nn.GELU())
        self.attribute_head = nn.Linear(hidden_size, len(ATTRIBUTES))
        self.format_head = nn.Linear(hidden_size, 3)

    def forward(self, image_features: torch.Tensor, text_tokens: torch.Tensor) -> dict:
        text_vector = self.text_embedding(text_tokens).mean(dim=1)
        image_vector = self.image_adapter(image_features)
        fused = self.fusion(torch.cat([text_vector, image_vector], dim=-1))
        return {"attribute_logits": self.attribute_head(fused), "format_logits": self.format_head(fused)}

    def extract_json(self, image_features: torch.Tensor, text_tokens: torch.Tensor, threshold: float = 0.5) -> list:
        with torch.no_grad():
            outputs = self.forward(image_features, text_tokens)
            active = torch.sigmoid(outputs["attribute_logits"]) >= threshold
        result = []
        for row in active:
            result.append({attribute: bool(row[index]) for index, attribute in enumerate(ATTRIBUTES)})
        return result
