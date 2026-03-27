from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast


@dataclass
class KARMAOutput:
    action_logit: torch.Tensor  # (B,)
    gen_logits: torch.Tensor  # (B,T,V)
    recon: torch.Tensor  # (B,d)


class CrossModalAligner(nn.Module):
    def __init__(self, d_model: int, nhead: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.behavior_to_text = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.behavior_to_image = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, behavior: torch.Tensor, text: torch.Tensor, image: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # attn_mask: (B,L) bool
        key_padding_mask = ~attn_mask

        bt, _ = self.behavior_to_text(behavior, text, text, key_padding_mask=key_padding_mask)
        bi, _ = self.behavior_to_image(behavior, image, image, key_padding_mask=key_padding_mask)
        fused = behavior + bt + bi
        return self.norm(fused)


class SemanticGenerator(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.gru = nn.GRU(input_size=d_model, hidden_size=hidden_dim, batch_first=True)
        self.init = nn.Linear(d_model, hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, interest: torch.Tensor, sem_in: torch.Tensor) -> torch.Tensor:
        # interest: (B,d) ; sem_in: (B,T)
        emb = self.embed(sem_in)
        h0 = torch.tanh(self.init(interest)).unsqueeze(0)  # (1,B,H)
        out, _ = self.gru(emb, h0)
        return self.head(out)


class OneModel(nn.Module):
    """KARMA: Knowledge-Action Regularized Multimodal Alignment (toy).

    Captures the two key elements described in the paper summary:
    1) Multimodal history alignment into a next-interest embedding.
    2) Semantic decodability via generation + reconstruction auxiliaries.
    """

    def __init__(
        self,
        d_model: int = 128,
        vocab_size: int = 3000,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )

        self.aligner = CrossModalAligner(d_model=d_model, nhead=nhead, dropout=dropout)
        self.interest_compressor = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.interest_proj = nn.Linear(d_model, d_model)

        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

        self.semantic_decoder = SemanticGenerator(d_model=d_model, vocab_size=vocab_size)
        self.reconstructor = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))

    def freeze_modules(self) -> None:
        return

    def get_optim(self, lr: float = 3e-4, weight_decay: float = 0.01) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

    def _masked_mean(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.unsqueeze(-1).type_as(x)
        denom = m.sum(dim=1).clamp_min(1.0)
        return (x * m).sum(dim=1) / denom

    @autocast()
    def forward(self, batch: Dict[str, torch.Tensor]) -> KARMAOutput:
        behavior = batch["behavior"]  # (B,L,d)
        text = batch["text"]
        image = batch["image"]
        attn_mask = batch["attn_mask"]  # (B,L)
        sem_ids = batch["sem_ids"]  # (B,T)

        aligned = self.aligner(behavior, text, image, attn_mask)
        compressed = self.interest_compressor(aligned, src_key_padding_mask=~attn_mask)
        interest = self._masked_mean(compressed, attn_mask)
        interest = self.interest_proj(interest)

        action_logit = self.action_head(interest).squeeze(-1)

        # Teacher forcing: use sem_ids as decoder input.
        gen_logits = self.semantic_decoder(interest, sem_ids)
        recon = self.reconstructor(interest)

        return KARMAOutput(action_logit=action_logit, gen_logits=gen_logits, recon=recon)


def karma_loss(
    out: KARMAOutput,
    action_label: torch.Tensor,
    sem_ids: torch.Tensor,
    sem_vec: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    action_loss = F.binary_cross_entropy_with_logits(out.action_logit, action_label)

    gen_loss = F.cross_entropy(out.gen_logits.reshape(-1, out.gen_logits.shape[-1]), sem_ids.reshape(-1))
    recon_loss = F.mse_loss(out.recon, sem_vec)

    total = action_loss + alpha * gen_loss + beta * recon_loss
    return total, {
        "action_loss": float(action_loss.detach()),
        "gen_loss": float(gen_loss.detach()),
        "recon_loss": float(recon_loss.detach()),
    }
