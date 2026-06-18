from __future__ import annotations

"""ProductConsistency — identity-preserving instruction-based product image editing.

Faithful *toy but runnable* reproduction of the method in:
  Khanna et al., "ProductConsistency: Improving Product Identity Preservation in
  Instruction-Based Image Editing via SFT and RL", arXiv:2606.19103.

The paper fine-tunes large diffusion editors (Qwen-Image-Edit / Flux.1-Kontext).
Those backbones are far too heavy for a self-contained reproduction, so here we keep
the *method logic* identical while shrinking every component:

  * ProductEditor  — an encoder/decoder image editor conditioned on an edit
                     instruction. For RL it is *stochastic*: a small policy head
                     emits a Gaussian over a background-style latent that is sampled
                     and decoded (this is the editable degree of freedom; the product
                     region is what we want preserved).
  * ProductCaptioner — the "re-caption / OCR" module used by the Cyclic Consistency
                     reward. It reads an image and predicts the product description
                     = (brand id, on-product text token sequence). This stands in for
                     the closed-source captioner + OCR used in the paper.
  * cyclic_consistency_reward — description -> edited image -> re-caption ->
                     similarity(original description, re-caption). This is the core
                     contribution: a computable semantic proxy for product identity
                     that is used as the RL reward (Eq. cyclic consistency in paper).

Interfaces (image tensor shapes, description dicts) are shared across data.py /
train.py / test.py so the toy pipeline is drop-in consistent with a real one.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- shared problem spec ------------------------------------------------------
IMG = 32                 # image side
PROD = 16                # product patch side (centered)
N_BRANDS = 8             # brand vocabulary (proxy for branding / logo identity)
TEXT_LEN = 4             # number of on-product text cells
N_CHARS = 10             # per-cell character vocabulary (digits 0-9), proxy for OCR
N_STYLES = 6             # number of background styles (the "edit" axis)


@dataclass(frozen=True)
class PCConfig:
    d: int = 64
    style_dim: int = 16
    n_instructions: int = N_STYLES   # instruction = "put product on background k"


# ---- editor ------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, ci, co, down=True):
        super().__init__()
        s = 2 if down else 1
        self.net = nn.Sequential(
            nn.Conv2d(ci, co, 3, s, 1), nn.GroupNorm(8, co), nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class ProductEditor(nn.Module):
    """Encoder/decoder editor conditioned on (instruction, style latent).

    SFT mode  : style latent comes from the editor's *mean* head (deterministic);
                trained to reconstruct the ground-truth edited target.
    RL  mode  : style latent is *sampled* from N(mu, sigma); the log-prob is returned
                for policy-gradient updates against the cyclic-consistency reward.
    """

    def __init__(self, cfg: PCConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d
        self.instr_emb = nn.Embedding(cfg.n_instructions, d)
        # encoder
        self.enc1 = ConvBlock(3, d, down=True)       # 32 -> 16
        self.enc2 = ConvBlock(d, d * 2, down=True)   # 16 -> 8
        # policy over background style latent (the editable d.o.f.)
        self.to_mu = nn.Linear(d * 2 * 8 * 8 + d, cfg.style_dim)
        self.to_logstd = nn.Linear(d * 2 * 8 * 8 + d, cfg.style_dim)
        # decoder (conditioned on style + instruction)
        self.style_proj = nn.Linear(cfg.style_dim + d, d * 2 * 8 * 8)
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(d * 4, d, 4, 2, 1), nn.GroupNorm(8, d), nn.SiLU())  # 8 -> 16
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(d * 2, d, 4, 2, 1), nn.GroupNorm(8, d), nn.SiLU())  # 16 -> 32
        self.out = nn.Conv2d(d, 3, 3, 1, 1)

    def encode(self, img, instr):
        c = self.instr_emb(instr)                       # [B, d]
        h1 = self.enc1(img)                             # [B, d, 16, 16]
        h2 = self.enc2(h1)                              # [B, 2d, 8, 8]
        flat = torch.cat([h2.flatten(1), c], dim=1)
        mu = self.to_mu(flat)
        logstd = self.to_logstd(flat).clamp(-4, 2)
        return c, h1, h2, mu, logstd

    def decode(self, c, h1, h2, style):
        d = self.cfg.d
        z = self.style_proj(torch.cat([style, c], dim=1)).view(-1, d * 2, 8, 8)
        x = self.dec2(torch.cat([h2, z], dim=1))        # [B, d, 16, 16]
        x = self.dec1(torch.cat([h1, x], dim=1))        # [B, d, 32, 32]
        return torch.sigmoid(self.out(x))

    def forward(self, img, instr, sample: bool = False):
        c, h1, h2, mu, logstd = self.encode(img, instr)
        if sample:
            std = logstd.exp()
            eps = torch.randn_like(std)
            style = mu + eps * std
            logp = (-0.5 * (((style - mu) / (std + 1e-6)) ** 2)
                    - logstd - 0.5 * torch.log(torch.tensor(2 * torch.pi))).sum(-1)
        else:
            style = mu
            logp = torch.zeros(img.size(0), device=img.device)
        edited = self.decode(c, h1, h2, style)
        return edited, logp


# ---- captioner / OCR (re-caption module) -------------------------------------
class ProductCaptioner(nn.Module):
    """Reads an image and predicts the product description (brand + text cells).

    Used both as an evaluation oracle and as the differentiable similarity backend
    of the Cyclic Consistency reward.
    """

    def __init__(self, cfg: PCConfig):
        super().__init__()
        d = cfg.d
        self.backbone = nn.Sequential(
            ConvBlock(3, d, True), ConvBlock(d, d * 2, True), ConvBlock(d * 2, d * 2, True),
        )                                                # 32 -> 4
        self.brand_head = nn.Linear(d * 2 * 4 * 4, N_BRANDS)
        self.text_head = nn.Linear(d * 2 * 4 * 4, TEXT_LEN * N_CHARS)

    def forward(self, img):
        h = self.backbone(img).flatten(1)
        brand = self.brand_head(h)                        # [B, N_BRANDS]
        text = self.text_head(h).view(-1, TEXT_LEN, N_CHARS)
        return brand, text


def caption_distribution(captioner, img):
    brand, text = captioner(img)
    return F.log_softmax(brand, -1), F.log_softmax(text, -1)


def cyclic_consistency_reward(captioner, edited, desc):
    """Core contribution: sim(original description, caption(edited image)).

    We use the captioner's *probability* assigned to the ORIGINAL brand + text
    tokens as the semantic-similarity proxy. Higher == product identity preserved.
    Returns a per-sample reward in roughly [0, 1].
    """
    log_brand, log_text = caption_distribution(captioner, edited)
    b = desc["brand"]
    t = desc["text"]                                      # [B, TEXT_LEN]
    r_brand = log_brand.gather(1, b.unsqueeze(1)).squeeze(1).exp()
    r_text = log_text.gather(2, t.unsqueeze(2)).squeeze(2).exp().mean(1)
    return 0.5 * r_brand + 0.5 * r_text
