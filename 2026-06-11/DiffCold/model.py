import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class DiffusionConfig:
    embed_dim: int = 64
    content_dim: int = 64
    hidden_dim: int = 256
    num_steps: int = 20
    beta_start: float = 1e-4
    beta_end: float = 0.02


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    # t: [B] int
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(0, half, device=t.device).float() / half)
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb


class NoisePredictor(nn.Module):
    def __init__(self, cfg: DiffusionConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.t_proj = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
        )

        in_dim = cfg.embed_dim + cfg.content_dim + cfg.embed_dim + cfg.hidden_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.embed_dim),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, content: torch.Tensor, start: torch.Tensor) -> torch.Tensor:
        t_emb = sinusoidal_time_embedding(t, self.cfg.hidden_dim)
        t_emb = self.t_proj(t_emb)
        h = torch.cat([x_t, content, start, t_emb], dim=-1)
        return self.net(h)


class DiffColdDDPM(nn.Module):
    def __init__(self, cfg: DiffusionConfig) -> None:
        super().__init__()
        self.cfg = cfg

        betas = torch.linspace(cfg.beta_start, cfg.beta_end, steps=cfg.num_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

        self.model = NoisePredictor(cfg)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x_t = sqrt(alpha_bar) x0 + sqrt(1-alpha_bar) noise
        a_bar = self.alphas_cumprod[t - 1].unsqueeze(1)
        return torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * noise

    def predict_x0(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        a_bar = self.alphas_cumprod[t - 1].unsqueeze(1)
        return (x_t - torch.sqrt(1.0 - a_bar) * eps) / torch.sqrt(a_bar + 1e-12)

    def p_sample(self, x_t: torch.Tensor, t: int, content: torch.Tensor, start: torch.Tensor) -> torch.Tensor:
        b = x_t.size(0)
        t_tensor = torch.full((b,), t, device=x_t.device, dtype=torch.long)
        eps = self.model(x_t, t_tensor, content, start)

        beta_t = self.betas[t - 1]
        alpha_t = self.alphas[t - 1]
        a_bar_t = self.alphas_cumprod[t - 1]

        x0_pred = (x_t - torch.sqrt(1.0 - a_bar_t) * eps) / torch.sqrt(a_bar_t + 1e-12)
        x0_pred = F.normalize(x0_pred, dim=-1)

        mean = (1.0 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1.0 - a_bar_t + 1e-12)) * eps)

        if t == 1:
            return F.normalize(mean, dim=-1)

        noise = torch.randn_like(x_t)
        sigma = torch.sqrt(beta_t)
        return mean + sigma * noise

    def sample(self, content: torch.Tensor, start: torch.Tensor, steps: int | None = None) -> torch.Tensor:
        steps = steps or self.cfg.num_steps
        x = start + 0.1 * torch.randn_like(start)

        for t in range(steps, 0, -1):
            x = self.p_sample(x, t, content, start)

        return F.normalize(x, dim=-1)

    def loss(self, x0: torch.Tensor, content: torch.Tensor, start: torch.Tensor) -> tuple[torch.Tensor, dict]:
        b = x0.size(0)
        t = torch.randint(1, self.cfg.num_steps + 1, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)

        eps_pred = self.model(x_t, t, content, start)
        ddpm_loss = F.mse_loss(eps_pred, noise)

        # alignment-style auxiliary loss (simplified): InfoNCE on predicted x0
        x0_pred = self.predict_x0(x_t, t, eps_pred)
        x0_pred = F.normalize(x0_pred, dim=-1)
        x0_gt = F.normalize(x0, dim=-1)

        logits = (x0_pred @ x0_gt.t()) / 0.07
        labels = torch.arange(b, device=x0.device)
        align_loss = F.cross_entropy(logits, labels)

        total = ddpm_loss + 0.2 * align_loss
        return total, {"ddpm": float(ddpm_loss.detach()), "align": float(align_loss.detach())}


class ContentProjector(nn.Module):
    def __init__(self, content_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(content_dim, embed_dim)

    def forward(self, content: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(content), dim=-1)
