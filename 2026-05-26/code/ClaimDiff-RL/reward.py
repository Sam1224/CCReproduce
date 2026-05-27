"""
ClaimDiff-RL Reward Composition.

Paper formulation (Section 3.2):
  Given per-difference statistics {n_hall, hall_sev, n_omit, omit_sev}:

  R_hall  = -α · Σ severity(d) for d in hallucinations
  R_omit  = -β · Σ severity(d) for d in omissions
  R_total = R_hall + R_omit + γ · length_bonus

  Key property: α and β are independently tunable, enabling researchers to
  navigate the Pareto frontier between reducing hallucinations and reducing omissions.

  When α >> β: model is penalized more for hallucinations → conservative, complete captions.
  When β >> α: model is penalized more for omissions    → verbose, covering captions.
"""
from dataclasses import dataclass
from typing import List
from judge import ClaimDiff


@dataclass
class RewardConfig:
    alpha: float = 1.0    # weight for hallucination penalty
    beta: float = 1.0     # weight for omission penalty
    gamma: float = 0.05   # small bonus for caption completeness (length relative to ref)
    max_length_ratio: float = 2.0  # clip length bonus at this ratio


def compute_reward(
    diffs: List[ClaimDiff],
    candidate: str,
    reference: str,
    config: RewardConfig = None,
) -> dict:
    """
    Compute the ClaimDiff-RL reward for a single (candidate, reference, diffs) triple.

    Returns a dict with component rewards and total reward.
    """
    if config is None:
        config = RewardConfig()

    hall_sev = sum(d.severity for d in diffs if d.is_hallucination)
    omit_sev = sum(d.severity for d in diffs if d.is_omission)

    R_hall = -config.alpha * hall_sev
    R_omit = -config.beta * omit_sev

    # Length bonus: small positive signal for candidates of similar length to reference
    cand_len = len(candidate.split())
    ref_len = max(len(reference.split()), 1)
    ratio = min(cand_len / ref_len, config.max_length_ratio)
    R_length = config.gamma * (1.0 - abs(1.0 - ratio))

    R_total = R_hall + R_omit + R_length

    return {
        "R_hallucination": R_hall,
        "R_omission": R_omit,
        "R_length": R_length,
        "R_total": R_total,
        "n_hall": sum(1 for d in diffs if d.is_hallucination),
        "n_omit": sum(1 for d in diffs if d.is_omission),
        "hall_severity": hall_sev,
        "omit_severity": omit_sev,
    }


def batch_rewards(
    candidates: List[str],
    references: List[str],
    diff_lists: List[List[ClaimDiff]],
    config: RewardConfig = None,
) -> List[dict]:
    """Compute rewards for a batch."""
    if config is None:
        config = RewardConfig()
    return [
        compute_reward(diffs, cand, ref, config)
        for cand, ref, diffs in zip(candidates, references, diff_lists)
    ]
