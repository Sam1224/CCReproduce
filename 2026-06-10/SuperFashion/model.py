from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelConfig:
    token_dim: int = 6
    model_dim: int = 128
    num_layers: int = 3
    num_heads: int = 4
    dropout: float = 0.1
    num_attrs: int = 3


class AttributeConditionedTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.attr_embed = nn.Embedding(cfg.num_attrs, cfg.model_dim)
        self.token_proj = nn.Linear(cfg.token_dim, cfg.model_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.model_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.model_dim * 4,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)
        self.out_norm = nn.LayerNorm(cfg.model_dim)

    def forward(self, tokens: torch.Tensor, attr_id: torch.Tensor) -> torch.Tensor:
        batch_size = tokens.shape[0]

        attr_tok = self.attr_embed(attr_id).unsqueeze(1)
        tok = self.token_proj(tokens)
        x = torch.cat([attr_tok, tok], dim=1)
        x = self.encoder(x)
        x = self.out_norm(x[:, 0])
        x = F.normalize(x, dim=-1)
        return x
