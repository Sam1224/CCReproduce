from __future__ import annotations

from dataclasses import dataclass

import torch

from data import ToyRecWorld


@dataclass
class RewardOutputs:
    # reward for each candidate
    reward: torch.FloatTensor  # [B, K]
    cand_item_id: torch.LongTensor  # [B, K]  (-1 means invalid)


class ToyRewardModel:
    """A toy 'production ranker' with exposure bias.

    We score a candidate by similarity to the (hidden) user preference.
    Since the toy dataset does not store user vectors, we proxy user preference
    with the target item's embedding (i.e., items similar to the target are good).

    Exposure bias is simulated by adding a popularity-correlated bias.
    """

    def __init__(self, world: ToyRecWorld, *, noise_std: float = 0.05) -> None:
        self.world = world
        self.noise_std = noise_std

    def seq_to_item_id(self, seq: torch.LongTensor) -> torch.LongTensor:
        # seq: [B, K, L]
        b, k, l = seq.shape
        out = torch.full((b, k), -1, dtype=torch.long, device=seq.device)
        for i in range(b):
            for j in range(k):
                key = tuple(int(x) for x in seq[i, j].tolist())
                if key in self.world.seq_to_item:
                    out[i, j] = self.world.seq_to_item[key]
        return out

    def score(self, *, target_item_id: torch.LongTensor, cand_seq: torch.LongTensor) -> RewardOutputs:
        device = cand_seq.device
        cand_item_id = self.seq_to_item_id(cand_seq)  # [B,K]

        # base similarity: item_emb[cand] dot item_emb[target]
        target_emb = self.world.item_emb[target_item_id].to(device)  # [B,D]

        b, k = cand_item_id.shape
        sim = torch.full((b, k), -1.0, dtype=torch.float32, device=device)
        valid_mask = cand_item_id >= 0
        if valid_mask.any():
            cand_emb = self.world.item_emb[cand_item_id.clamp(min=0)].to(device)  # [B,K,D]
            sim_valid = (cand_emb * target_emb.unsqueeze(1)).sum(dim=-1)  # [B,K]
            sim = torch.where(valid_mask, sim_valid, sim)

        bias = self.world.rm_bias.to(device)[cand_item_id.clamp(min=0)]
        bias = torch.where(valid_mask, bias, torch.zeros_like(bias))

        noise = torch.randn((b, k), device=device) * self.noise_std
        reward = sim + bias + noise
        reward = torch.where(valid_mask, reward, torch.full_like(reward, -2.0))

        return RewardOutputs(reward=reward, cand_item_id=cand_item_id)


def reward_discriminability(
    *,
    reward: torch.FloatTensor,
    target_in_candidates: torch.BoolTensor,
    gt_reward: torch.FloatTensor,
    margin: float = 0.2,
) -> torch.BoolTensor:
    """Return per-sample boolean whether reward model is discriminative.

    A simple proxy: GT must be present in candidates; and GT reward must exceed
    the max negative by at least margin.
    """

    # reward: [B,K]
    max_neg = reward.masked_fill(target_in_candidates, float("-inf")).max(dim=1).values
    ok = target_in_candidates.any(dim=1) & (gt_reward > max_neg + margin)
    return ok
