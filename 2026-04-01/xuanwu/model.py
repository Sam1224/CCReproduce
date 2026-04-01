from __future__ import annotations

import torch
import torch.nn as nn


class XuanwuToyMMClassifier(nn.Module):
    """A minimal multimodal classifier (text + image feature).

    Designed as a small placeholder reproduction for the paper:
    "Xuanwu: Evolving General Multimodal Models into an Industrial-Grade Foundation for Content Ecosystems".

    The real paper is a foundation-model system; here we implement a runnable
    multimodal fusion classifier with a similar *interface*.
    """

    def __init__(
        self,
        *,
        vocab_size: int = 2048,
        text_dim: int = 256,
        image_dim: int = 512,
        hidden_dim: int = 512,
        num_labels: int = 2,
    ) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, text_dim)
        self.image_proj = nn.Linear(image_dim, text_dim)

        self.fuse = nn.Sequential(
            nn.Linear(text_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(hidden_dim, num_labels)

    def forward(self, *, image_feat: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: [B, T]
        text = self.token_emb(token_ids).mean(dim=1)  # [B, D]
        img = self.image_proj(image_feat)  # [B, D]
        fused = self.fuse(torch.cat([text, img], dim=-1))
        return self.classifier(fused)
