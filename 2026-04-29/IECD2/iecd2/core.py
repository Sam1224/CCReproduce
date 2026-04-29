from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class IECDConfig:
    temperature_instruction: float = 1.0
    temperature_evidence: float = 0.9
    eta: float = -3.0
    eps: float = 1e-8


@dataclass(frozen=True)
class IECDStepResult:
    fused_probs: torch.Tensor
    gate_g: torch.Tensor
    symmetric_kl: torch.Tensor


def _probs_from_logits(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    return F.softmax(logits / temperature, dim=-1)


def _symmetric_kl(p: torch.Tensor, q: torch.Tensor, eps: float) -> torch.Tensor:
    if eps <= 0:
        raise ValueError("eps must be > 0")

    p = p.clamp_min(eps)
    q = q.clamp_min(eps)

    kl_pq = torch.sum(p * (p.log() - q.log()), dim=-1)
    kl_qp = torch.sum(q * (q.log() - p.log()), dim=-1)
    return kl_pq + kl_qp


def iecd_fuse_logits(
    logits_instruction: torch.Tensor,
    logits_evidence: torch.Tensor,
    config: IECDConfig = IECDConfig(),
) -> IECDStepResult:
    """Fuse two token distributions using IECD[2].

    Args:
        logits_instruction: [V] or [B, V] tensor, next-token logits under the instruction prompt.
        logits_evidence:    [V] or [B, V] tensor, next-token logits under the evidence prompt.
        config:             IECDConfig.

    Returns:
        IECDStepResult with fused_probs (same shape as logits), gating coefficient g, and symmetric KL.

    Notes:
        Paper formulation (Eq. 3-6):
        - Compute token distributions p_I and p_E.
        - Symmetric KL divergence D = KL(p_I||p_E) + KL(p_E||p_I).
        - Gate g = exp(eta * D) / (1 + exp(eta * D)) = sigmoid(eta * D).
        - Fuse with geometric mean: p ∝ p_I^g * p_E^(1-g).

        With eta < 0 (as used in the paper), high disagreement (large D) makes g -> 0, thus relying
        more on evidence stream.
    """

    if logits_instruction.shape != logits_evidence.shape:
        raise ValueError(
            f"shape mismatch: logits_instruction={logits_instruction.shape}, logits_evidence={logits_evidence.shape}"
        )

    p_i = _probs_from_logits(logits_instruction, config.temperature_instruction)
    p_e = _probs_from_logits(logits_evidence, config.temperature_evidence)

    d = _symmetric_kl(p_i, p_e, config.eps)
    g = torch.sigmoid(d * config.eta)

    fused_logp = g.unsqueeze(-1) * p_i.log() + (1.0 - g).unsqueeze(-1) * p_e.log()
    fused_probs = F.softmax(fused_logp, dim=-1)

    return IECDStepResult(fused_probs=fused_probs, gate_g=g, symmetric_kl=d)
