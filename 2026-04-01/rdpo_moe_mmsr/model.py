from __future__ import annotations

import torch
import torch.nn as nn


class Top1MoE(nn.Module):
    def __init__(self, *, dim: int, num_experts: int = 4, hidden: int = 256) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, dim)) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,L,D]
        logits = self.gate(x)  # [B,L,E]
        top = logits.argmax(dim=-1)  # [B,L]
        out = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):
            mask = top == e
            if mask.any():
                y = expert(x[mask])
                out[mask] = y
        return out


class RDPO_MoE_ToyModel(nn.Module):
    """Toy multimodal sequential recommender with sparse MoE.

    We use a small Transformer encoder; FFN is replaced by a Top-1 MoE.
    Training uses a DPO-like pairwise objective on (pos_next, neg_next).
    """

    def __init__(
        self,
        *,
        num_items: int = 8000,
        num_modalities: int = 3,
        dim: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.item_emb = nn.Embedding(num_items, dim)
        self.mod_emb = nn.Embedding(num_modalities, dim)
        self.pos_emb = nn.Embedding(256, dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=nhead, dim_feedforward=dim * 4, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.moe = Top1MoE(dim=dim)
        self.norm = nn.LayerNorm(dim)

    def encode(self, seq_item_ids: torch.Tensor, seq_modality_ids: torch.Tensor) -> torch.Tensor:
        b, l = seq_item_ids.shape
        pos = torch.arange(l, device=seq_item_ids.device).unsqueeze(0).expand(b, l)
        x = self.item_emb(seq_item_ids) + self.mod_emb(seq_modality_ids) + self.pos_emb(pos)
        x = self.encoder(x)
        x = self.norm(x + self.moe(x))
        return x[:, -1, :]  # last state

    def score_item(self, user_state: torch.Tensor, item_id: torch.Tensor) -> torch.Tensor:
        item = self.item_emb(item_id)
        return (user_state * item).sum(dim=-1)

    def forward(self, *, seq_item_ids: torch.Tensor, seq_modality_ids: torch.Tensor, pos_next: torch.Tensor, neg_next: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        u = self.encode(seq_item_ids, seq_modality_ids)
        return self.score_item(u, pos_next), self.score_item(u, neg_next)
