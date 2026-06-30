from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DualEncoderConfig:
    embed_dim: int = 128
    vocab_size: int = 128
    max_len: int = 32


class TinyTextEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, input_ids: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # input_ids: (B, L)
        x = self.emb(input_ids)  # (B, L, D)
        m = attn_mask.unsqueeze(-1).float()
        x = (x * m).sum(dim=1) / (m.sum(dim=1).clamp_min(1.0))
        x = self.proj(x)
        return x


class TinyImageEncoder(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(128, embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.net(images).flatten(1)
        return self.proj(x)


class DualEncoder(nn.Module):
    def __init__(self, cfg: DualEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.image = TinyImageEncoder(embed_dim=cfg.embed_dim)
        self.text = TinyTextEncoder(vocab_size=cfg.vocab_size, embed_dim=cfg.embed_dim)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        z = self.image(images)
        return F.normalize(z, dim=-1)

    def encode_text(self, input_ids: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        z = self.text(input_ids, attn_mask)
        return F.normalize(z, dim=-1)

    def forward(self, images: torch.Tensor, input_ids: torch.Tensor, attn_mask: torch.Tensor):
        zi = self.encode_image(images)
        zt = self.encode_text(input_ids, attn_mask)
        scale = self.logit_scale.exp().clamp(1.0, 100.0)
        logits = scale * (zi @ zt.t())
        return logits


def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    """Symmetric InfoNCE on the diagonal."""
    bsz = logits.size(0)
    labels = torch.arange(bsz, device=logits.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)
    return (loss_i2t + loss_t2i) / 2.0


def distill_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Row-wise KL(teacher || student) and column-wise KL."""
    t = float(temperature)

    s_row = F.log_softmax(student_logits / t, dim=1)
    t_row = F.softmax(teacher_logits / t, dim=1)
    kl_row = F.kl_div(s_row, t_row, reduction="batchmean") * (t * t)

    s_col = F.log_softmax(student_logits.t() / t, dim=1)
    t_col = F.softmax(teacher_logits.t() / t, dim=1)
    kl_col = F.kl_div(s_col, t_col, reduction="batchmean") * (t * t)

    return (kl_row + kl_col) / 2.0


def wise_ft_interpolate(
    base_state: dict[str, torch.Tensor],
    finetuned_state: dict[str, torch.Tensor],
    alpha: float,
) -> dict[str, torch.Tensor]:
    """WiSE-FT style weight interpolation.

    Parameters
    - alpha=0 -> base
    - alpha=1 -> finetuned
    """

    out: dict[str, torch.Tensor] = {}
    for k, v in finetuned_state.items():
        if k not in base_state:
            out[k] = v
            continue
        out[k] = (1 - alpha) * base_state[k] + alpha * v
    return out
