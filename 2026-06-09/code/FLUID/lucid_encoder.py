"""
LUCID: Cross-domain Multimodal Encoder with Hierarchical Discrete Codes.

Implements the core contribution of FLUID:
- Cross-modal fusion of visual, audio, and text features
- Residual Quantization (RQ) to produce hierarchical discrete codes (LUCID codes)
- These codes replace item IDs in the downstream ranker

Paper formula (Section 3.1):
    z = Encoder(v, a, t)           # continuous multimodal embedding
    c_1, c_2, ..., c_K = RQ(z)    # K-level discrete codes via Residual VQ
    where c_k = argmin_{e in E_k} ||r_{k-1} - e||^2
          r_k = r_{k-1} - c_k     # residual update
          r_0 = z
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class VectorQuantizer(nn.Module):
    """Single-level vector quantizer (VQ-VAE codebook)."""

    def __init__(self, codebook_size: int, embedding_dim: int):
        super().__init__()
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.codebook = nn.Embedding(codebook_size, embedding_dim)
        nn.init.uniform_(self.codebook.weight, -1 / codebook_size, 1 / codebook_size)

    def forward(self, z: torch.Tensor):
        """
        Args:
            z: (B, D) continuous embeddings
        Returns:
            z_q: (B, D) quantized embeddings
            codes: (B,) discrete codes (indices into codebook)
            residual: (B, D) residual = z - z_q
            loss: commitment + codebook loss
        """
        # distances from z to codebook entries: (B, codebook_size)
        distances = (
            z.pow(2).sum(-1, keepdim=True)
            - 2 * z @ self.codebook.weight.T
            + self.codebook.weight.pow(2).sum(-1)
        )
        codes = distances.argmin(-1)                       # (B,)
        z_q = self.codebook(codes)                         # (B, D)

        # Straight-through estimator
        z_q_st = z + (z_q - z).detach()

        residual = z - z_q.detach()                        # (B, D)

        # Commitment loss + codebook loss (eq. from VQ-VAE)
        loss = F.mse_loss(z_q, z.detach()) + 0.25 * F.mse_loss(z, z_q.detach())

        return z_q_st, codes, residual, loss


class ResidualQuantizer(nn.Module):
    """
    Residual Vector Quantizer (RQ-VAE) producing K-level hierarchical codes.

    For FLUID, K levels of discrete codes (LUCID codes) replace the item ID.
    """

    def __init__(self, n_levels: int, codebook_size: int, embedding_dim: int):
        super().__init__()
        self.n_levels = n_levels
        self.vqs = nn.ModuleList(
            [VectorQuantizer(codebook_size, embedding_dim) for _ in range(n_levels)]
        )

    def forward(self, z: torch.Tensor):
        """
        Args:
            z: (B, D) input embeddings
        Returns:
            codes_all: (B, K) discrete code indices at each level
            vq_loss: scalar total quantization loss
            z_reconstructed: (B, D) sum of all quantized vectors
        """
        residual = z
        codes_all = []
        vq_loss = 0.0
        z_reconstructed = torch.zeros_like(z)

        for vq in self.vqs:
            z_q, codes, residual, loss = vq(residual)
            codes_all.append(codes)
            vq_loss = vq_loss + loss
            z_reconstructed = z_reconstructed + z_q

        codes_all = torch.stack(codes_all, dim=1)          # (B, K)
        return codes_all, vq_loss, z_reconstructed


class LUCIDEncoder(nn.Module):
    """
    Cross-domain Multimodal Encoder (LUCID) from FLUID.

    Fuses visual, audio, and text features into a shared embedding,
    then produces hierarchical discrete codes via Residual VQ.

    This is the core module that produces LUCID codes replacing item IDs.
    """

    def __init__(
        self,
        visual_dim: int = 256,
        audio_dim: int = 128,
        text_dim: int = 128,
        hidden_dim: int = 256,
        n_levels: int = 4,          # K levels of hierarchy (paper: K=4)
        codebook_size: int = 512,   # size of each VQ codebook
    ):
        super().__init__()
        # Project each modality to shared hidden dim
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        # Cross-modal fusion (cross-attention inspired, simplified)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # Residual Quantizer producing hierarchical LUCID codes
        self.rq = ResidualQuantizer(n_levels, codebook_size, hidden_dim)

        self.hidden_dim = hidden_dim
        self.n_levels = n_levels
        self.codebook_size = codebook_size

    def encode(self, visual: torch.Tensor, audio: torch.Tensor, text: torch.Tensor):
        """Produce continuous embedding z from multimodal inputs."""
        v = F.normalize(self.visual_proj(visual), dim=-1)
        a = F.normalize(self.audio_proj(audio), dim=-1)
        t = F.normalize(self.text_proj(text), dim=-1)
        z = self.fusion(torch.cat([v, a, t], dim=-1))
        return z

    def forward(self, visual: torch.Tensor, audio: torch.Tensor, text: torch.Tensor):
        """
        Args:
            visual: (B, visual_dim)
            audio:  (B, audio_dim)
            text:   (B, text_dim)
        Returns:
            codes:  (B, K) hierarchical discrete codes (LUCID codes)
            vq_loss: scalar
            z_q:    (B, D) reconstructed embedding
        """
        z = self.encode(visual, audio, text)
        codes, vq_loss, z_q = self.rq(z)
        return codes, vq_loss, z_q
