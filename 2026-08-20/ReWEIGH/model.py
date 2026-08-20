from typing import Dict

import torch
from torch import nn


class ToyLVLM(nn.Module):
    def __init__(self, vocab_size: int, image_dim: int = 12, hidden_dim: int = 64, visual_tokens: int = 6):
        super().__init__()
        self.vocab_size = vocab_size
        self.visual_tokens = visual_tokens
        self.visual_encoder = nn.Sequential(
            nn.Linear(image_dim, hidden_dim * visual_tokens),
            nn.Tanh(),
        )
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.context = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.popularity_bias = nn.Parameter(torch.zeros(vocab_size))

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        visual = self.visual_encoder(image)
        return visual.view(image.size(0), self.visual_tokens, -1)

    def visual_readout(self, visual_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(visual_hidden_states)

    def forward(self, image: torch.Tensor, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        visual_hidden = self.encode_image(image)
        token_hidden, _ = self.context(self.token_embedding(input_ids))
        pooled_visual = visual_hidden.mean(dim=1, keepdim=True).expand(-1, token_hidden.size(1), -1)
        fused = torch.tanh(self.fusion(torch.cat([token_hidden, pooled_visual], dim=-1)))
        logits = self.lm_head(fused) + self.popularity_bias
        return {"logits": logits, "visual_hidden_states": visual_hidden}


def build_toy_model(vocab_size: int) -> ToyLVLM:
    model = ToyLVLM(vocab_size=vocab_size)
    with torch.no_grad():
        model.popularity_bias.zero_()
        for idx in [4, 5, 6, 7, 8, 9]:
            if idx < vocab_size:
                model.popularity_bias[idx] = 0.35
    return model
