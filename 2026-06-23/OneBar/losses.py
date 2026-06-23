"""
losses.py — OneBar training objectives.

Covers the three progressive-preference-internalization stages (paper Sec.3.4):

  Stage 1  SFT cross-entropy            (Eq. 5)
  Stage 2  list-wise Softmax DPO        (Eq. 9 / Eq. 10)
  Stage 3  PIOPD = SFT + FKL + RKL      (Eq. 13 / Eq. 16) + entropy reg / R-Drop

All functions are pure-tensor and unit-test friendly.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Stage 1 : SFT cross-entropy  (Eq. 5)
# --------------------------------------------------------------------------- #
def sft_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Standard next-token CE; labels use -100 for ignore (pad / [REJECT] pad).

    L_SFT = - E sum_t log p_theta(y_t | y_<t, s_x)
    `logits` : [B, T, V] decoder logits aligned with `labels` [B, T].
    """
    V = logits.size(-1)
    return F.cross_entropy(logits.view(-1, V), labels.view(-1), ignore_index=-100)


# --------------------------------------------------------------------------- #
# Stage 2 : list-wise Softmax DPO  (Eq. 9 / Eq. 10)
# --------------------------------------------------------------------------- #
def listwise_dpo_loss(policy_logps: torch.Tensor,
                      ref_logps: torch.Tensor,
                      levels: torch.Tensor,
                      beta: float = 0.1,
                      temperature: float = 1.0,
                      lambda_sft: float = 1.0) -> torch.Tensor:
    """List-wise Softmax DPO over a behavior-ranked candidate list.

    Args (all for ONE trigger's candidate list, sorted best-first by level):
      policy_logps : [n]  sequence log-probs l_theta(y_i | x)      (Eq. 6)
      ref_logps    : [n]  reference (frozen) sequence log-probs
      levels       : [n]  behavior level b(y_i) in {1..6} (1 best)
      beta, temperature, lambda_sft : Eq. 10 hyper-params (0.1 / 1.0 / 1.0)

    Implements Eq. 10:
      L = sum_{j in C_x} [ log(1 + sum_{i>j} exp((beta/T)(r_i - r_j)))
                           - lambda_sft * l_theta(y_j | x) ]
    where r_i = l_theta(y_i) - l_ref(y_i), and clicked anchors
      C_x = { j | b(y_j) <= 4, j < n }.
    """
    r = policy_logps - ref_logps                       # [n]  log-ratios
    n = r.size(0)
    scale = beta / temperature

    total = policy_logps.new_zeros(())
    n_anchor = 0
    for j in range(n):
        # clicked anchor condition: level <= 4 and has lower-ranked suffix
        if levels[j].item() <= 4 and j < n - 1:
            diffs = scale * (r[j + 1:] - r[j])          # r_i - r_j for i>j
            # log(1 + sum exp(diffs)) == logsumexp([0, diffs])
            zeros = diffs.new_zeros(1)
            nll = torch.logsumexp(torch.cat([zeros, diffs]), dim=0)
            sft_anchor = lambda_sft * policy_logps[j]
            total = total + (nll - sft_anchor)
            n_anchor += 1
    if n_anchor == 0:
        # no clicked anchor with a suffix -> fall back to a mild SFT pull on best
        return -lambda_sft * policy_logps[0] * 0.0 + policy_logps.sum() * 0.0
    return total / n_anchor


# --------------------------------------------------------------------------- #
# Stage 3 : PIOPD distillation losses  (Eq. 13)
# --------------------------------------------------------------------------- #
def _masked_token_kl(p: torch.Tensor, q: torch.Tensor, mask: torch.Tensor):
    """KL(p || q) summed over vocab, averaged over masked (valid) tokens.

    p, q : [B, T, V] probability distributions.
    mask : [B, T] float (1 for valid token positions).
    """
    kl = (p * (torch.log(p + 1e-9) - torch.log(q + 1e-9))).sum(-1)  # [B,T]
    denom = mask.sum().clamp_min(1.0)
    return (kl * mask).sum() / denom


def distill_loss(student_logits: torch.Tensor,
                 teacher_logits: torch.Tensor,
                 mask: torch.Tensor,
                 tau: float = 2.0,
                 lambda_fkl: float = 1.0,
                 lambda_rkl: float = 1.0):
    """Bidirectional KL distillation (Eq. 13).

      p_S      = softmax(z_S)
      p_T,tau  = softmax(z_T / tau)
      L_distill = tau^2 * [ lambda_fkl * KL(p_T || p_S)      (forward KL)
                          + lambda_rkl * KL(p_S || p_T) ]    (reverse KL)

    Forward KL makes the student COVER teacher-supported alternatives;
    reverse KL SUPPRESSES student mass on teacher-unsupported tokens.

    student_logits/teacher_logits : [B, T, V] aligned over the rollout prefix.
    Returns (total, fkl, rkl).
    """
    p_S = F.softmax(student_logits, dim=-1)
    p_T = F.softmax(teacher_logits / tau, dim=-1)
    fkl = _masked_token_kl(p_T, p_S, mask)             # KL(p_T || p_S)
    rkl = _masked_token_kl(p_S, p_T, mask)             # KL(p_S || p_T)
    total = (tau ** 2) * (lambda_fkl * fkl + lambda_rkl * rkl)
    return total, fkl.detach(), rkl.detach()


def entropy_regularizer(student_logits: torch.Tensor, mask: torch.Tensor):
    """L_Ent = - E sum_t H(p_S)  (Eq. 14, negative entropy).

    Minimising with a POSITIVE weight encourages higher student entropy and
    prevents the over-concentration / top-1 collapse described in Fig. 4.
    Returns the negative-entropy value (to be ADDED with +lambda_E).
    """
    p_S = F.softmax(student_logits, dim=-1)
    logp = F.log_softmax(student_logits, dim=-1)
    neg_ent = (p_S * logp).sum(-1)                     # = -H(p_S)  [B,T]
    denom = mask.sum().clamp_min(1.0)
    return (neg_ent * mask).sum() / denom


def rdrop_loss(logits1: torch.Tensor, logits2: torch.Tensor, mask: torch.Tensor):
    """R-Drop symmetric KL between two dropout passes (Eq. 15).

      L_RDrop = 0.5 * ( KL(p1 || p2) + KL(p2 || p1) )
    """
    p1 = F.softmax(logits1, dim=-1)
    p2 = F.softmax(logits2, dim=-1)
    kl12 = _masked_token_kl(p1, p2, mask)
    kl21 = _masked_token_kl(p2, p1, mask)
    return 0.5 * (kl12 + kl21)


def piopd_loss(student_label_logits, labels,
               student_rollout_logits, teacher_rollout_logits, rollout_mask,
               student_rollout_logits_2=None,
               tau: float = 2.0,
               lambda_fkl: float = 1.0, lambda_rkl: float = 1.0,
               lambda_ent: float = 0.01, lambda_rdrop: float = 0.0):
    """Combined PIOPD objective (Eq. 16):

        L_PIOPD = L_SFT + L_Distill + lambda_E * L_Ent
                  + lambda_RD * L_RDrop   (+ FGM handled in train loop)

    Components:
      * L_SFT     : CE on the clicked target y_ref (ground-truth anchor)
      * L_Distill : FKL + RKL between teacher (posterior prompt) & student,
                    evaluated on the student's ON-POLICY rollout prefix.
      * L_Ent     : entropy regulariser (anti-collapse)
      * L_RDrop   : optional output-space consistency (needs a 2nd pass)

    Returns (total, dict_of_components).
    """
    l_sft = sft_loss(student_label_logits, labels)
    l_distill, fkl, rkl = distill_loss(
        student_rollout_logits, teacher_rollout_logits, rollout_mask,
        tau=tau, lambda_fkl=lambda_fkl, lambda_rkl=lambda_rkl)
    l_ent = entropy_regularizer(student_rollout_logits, rollout_mask)

    total = l_sft + l_distill + lambda_ent * l_ent
    comp = {"sft": l_sft.detach(), "distill": l_distill.detach(),
            "fkl": fkl, "rkl": rkl, "ent": l_ent.detach()}

    if lambda_rdrop > 0 and student_rollout_logits_2 is not None:
        l_rd = rdrop_loss(student_rollout_logits, student_rollout_logits_2,
                          rollout_mask)
        total = total + lambda_rdrop * l_rd
        comp["rdrop"] = l_rd.detach()

    return total, comp
