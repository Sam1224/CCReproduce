from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from policy import DigitPolicy
from toy_task import feasible_prefix, verifiable_reward


@dataclass
class Rollout:
    tokens: List[int]
    logp: float
    reward: float
    pruned: bool


def rollout_sampling(
    policy: DigitPolicy,
    target: int,
    length: int,
    n_rollouts: int,
    device: str,
    seed: int,
) -> Tuple[List[Rollout], int]:
    """Baseline: sample full sequences (no pruning)."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(int(seed))

    tokens_sampled = 0
    outs: List[Rollout] = []

    for _ in range(n_rollouts):
        seq: List[int] = []
        logp = 0.0

        # start token: pick random digit to seed
        prev = torch.tensor([[int(rng.integers(0, 10))]], device=device)
        tgt = torch.tensor([[float(target)]], device=device)
        seq.append(int(prev.item()))
        tokens_sampled += 1

        for t in range(1, length):
            logits = policy(tgt, prev)
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            lp = torch.log(probs.gather(1, nxt) + 1e-12).item()
            logp += lp
            prev = torch.cat([prev, nxt], dim=1)
            seq.append(int(nxt.item()))
            tokens_sampled += 1

        r = verifiable_reward(seq, target)
        outs.append(Rollout(tokens=seq, logp=logp, reward=r, pruned=False))

    return outs, tokens_sampled


def rollout_pruned_sampling(
    policy: DigitPolicy,
    target: int,
    length: int,
    n_rollouts: int,
    device: str,
    seed: int,
) -> Tuple[List[Rollout], int]:
    """Online rollout pruning during generation.

    We sample rollouts token-by-token, and **stop early** for sequences that can no
    longer reach the target (feasibility pruning).

    Compute proxy: number of tokens sampled.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(int(seed))

    tokens_sampled = 0
    outs: List[Rollout] = []

    for _ in range(n_rollouts):
        seq: List[int] = []
        logp = 0.0
        prefix_sum = 0
        pruned = False

        prev = torch.tensor([[int(rng.integers(0, 10))]], device=device)
        tgt = torch.tensor([[float(target)]], device=device)
        first = int(prev.item())
        seq.append(first)
        prefix_sum += first
        tokens_sampled += 1

        if not feasible_prefix(prefix_sum, steps_done=1, length=length, target=target):
            outs.append(Rollout(tokens=seq, logp=logp, reward=0.0, pruned=True))
            continue

        for t in range(1, length):
            logits = policy(tgt, prev)
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            lp = torch.log(probs.gather(1, nxt) + 1e-12).item()
            logp += lp

            prev = torch.cat([prev, nxt], dim=1)
            dig = int(nxt.item())
            seq.append(dig)
            prefix_sum += dig
            tokens_sampled += 1

            if not feasible_prefix(prefix_sum, steps_done=t + 1, length=length, target=target):
                pruned = True
                break

        r = 0.0 if pruned else verifiable_reward(seq, target)
        outs.append(Rollout(tokens=seq, logp=logp, reward=r, pruned=pruned))

    return outs, tokens_sampled
