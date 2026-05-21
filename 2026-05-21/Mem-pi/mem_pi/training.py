from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import Task, Vocab
from .env import run_episode
from .model import MemPiPolicy


@dataclass
class TrainConfig:
    seed: int = 0
    device: str = "cpu"
    batch_size: int = 64
    lr: float = 3e-4
    max_len: int = 16


def collate_tasks(vocab: Vocab, tasks: Sequence[Task], max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    encoded = [vocab.encode(t.context_tokens)[:max_len] for t in tasks]
    t = max(len(x) for x in encoded)
    t = min(t, max_len)
    token_ids = torch.full((len(tasks), t), vocab.pad_id, dtype=torch.long)
    mask = torch.zeros((len(tasks), t), dtype=torch.bool)

    for i, ids in enumerate(encoded):
        ids = ids[:t]
        token_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        mask[i, : len(ids)] = True

    return token_ids, mask


def train_stage1_experience_distillation(
    *,
    policy: MemPiPolicy,
    vocab: Vocab,
    experience_bank: Sequence[Tuple[Task, int]],
    cfg: TrainConfig,
    epochs: int = 10,
) -> MemPiPolicy:
    """Stage 1: supervised distillation from an offline experience bank.

    In the paper this corresponds to compressing reusable experience into the
    parameters of pi_mem.

    Here we train the hint head to predict a discrete hint id (tool_id).
    """

    rng = np.random.default_rng(cfg.seed)
    device = torch.device(cfg.device)
    policy.to(device)
    policy.train()

    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss()

    indices = np.arange(len(experience_bank))
    for _ in range(epochs):
        rng.shuffle(indices)
        for start in range(0, len(indices), cfg.batch_size):
            batch_idx = indices[start : start + cfg.batch_size]
            batch = [experience_bank[i] for i in batch_idx]
            tasks = [t for t, _ in batch]
            targets = torch.tensor([hint_id for _, hint_id in batch], dtype=torch.long, device=device)

            token_ids, mask = collate_tasks(vocab, tasks, cfg.max_len)
            token_ids = token_ids.to(device)
            mask = mask.to(device)

            out = policy(token_ids, mask)
            loss = loss_fn(out.hint_logits, targets)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    return policy


def train_stage2_adaptation_distillation(
    *,
    policy: MemPiPolicy,
    vocab: Vocab,
    tasks: Sequence[Task],
    cfg: TrainConfig,
    steps: int = 800,
    group_size: int = 4,
    length_penalty: float = 0.3,
) -> MemPiPolicy:
    """Stage 2: decision-content decoupled policy optimization (toy).

    The paper uses GRPO with a structured rollout group: one ABSTAIN branch and
    multiple GENERATE branches. Then it decomposes the learning signal into
    decision-level and content-level components.

    We implement a lightweight analogue:

    - r0: reward when ABSTAIN (downstream agent runs with no memory)
    - rj: rewards for multiple sampled memory hints (GENERATE)

    decision_adv = mean(r_generate) - r0
    content_adv_j = rj - mean(r_generate)

    Loss:
    - maximize log P(GENERATE) weighted by decision_adv
    - maximize log P(hint_id) weighted by content_adv_j

    This is sufficient to reproduce the key behavior: learn to abstain on tasks
    solvable by the base agent, and generate concise, useful hints otherwise.
    """

    assert group_size >= 2

    rng = np.random.default_rng(cfg.seed)
    device = torch.device(cfg.device)
    policy.to(device)
    policy.train()

    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.lr)

    tasks_list = list(tasks)

    for _ in tqdm(range(steps), desc="stage2", leave=False):
        batch = [tasks_list[int(rng.integers(0, len(tasks_list)))] for _ in range(cfg.batch_size)]

        token_ids, mask = collate_tasks(vocab, batch, cfg.max_len)
        token_ids = token_ids.to(device)
        mask = mask.to(device)

        out = policy(token_ids, mask)
        decision_probs = torch.softmax(out.decision_logits, dim=-1)
        p_generate = decision_probs[:, 1].clamp_min(1e-6)
        log_p_generate = torch.log(p_generate)

        # Counterfactual abstain reward (no memory injected)
        abstain_rewards = []
        for t in batch:
            res = run_episode(task=t, rng=np.random.default_rng(0), hint_tool_id=None)
            abstain_rewards.append(1.0 if res.success else 0.0)
        r0 = torch.tensor(abstain_rewards, dtype=torch.float32, device=device)

        # Sample multiple memory contents (generate branches)
        gen_rewards: List[torch.Tensor] = []
        gen_logps: List[torch.Tensor] = []
        for _ in range(group_size - 1):
            hint_id, hint_logp = policy.sample_generate_hint(out.hint_logits)
            rewards = []
            for t, hid in zip(batch, hint_id.tolist()):
                res = run_episode(task=t, rng=np.random.default_rng(0), hint_tool_id=hid)
                rew = (1.0 if res.success else 0.0) - length_penalty
                rewards.append(rew)
            gen_rewards.append(torch.tensor(rewards, dtype=torch.float32, device=device))
            gen_logps.append(hint_logp)

        r_gen = torch.stack(gen_rewards, dim=0)  # (G-1, B)
        logp_hint = torch.stack(gen_logps, dim=0)  # (G-1, B)

        r_gen_mean = r_gen.mean(dim=0)
        decision_adv = (r_gen_mean - r0).detach()

        content_adv = (r_gen - r_gen_mean.unsqueeze(0)).detach()

        loss_decision = -(decision_adv * log_p_generate).mean()
        loss_content = -(content_adv * logp_hint).mean()

        loss = loss_decision + loss_content

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return policy


@torch.no_grad()
def evaluate(
    *,
    policy: MemPiPolicy,
    vocab: Vocab,
    tasks: Sequence[Task],
    seed: int,
) -> dict:
    policy.eval()
    device = next(policy.parameters()).device

    token_ids, mask = collate_tasks(vocab, tasks, max_len=16)
    token_ids = token_ids.to(device)
    mask = mask.to(device)

    out = policy(token_ids, mask)
    p_generate = torch.softmax(out.decision_logits, dim=-1)[:, 1]
    hint_id = torch.argmax(out.hint_logits, dim=-1)

    rng = np.random.default_rng(seed)

    base_succ = 0
    always_gen_succ = 0
    adaptive_succ = 0

    for t, pg, hid in zip(tasks, p_generate.tolist(), hint_id.tolist()):
        base = run_episode(task=t, rng=rng, hint_tool_id=None)
        base_succ += int(base.success)

        gen = run_episode(task=t, rng=rng, hint_tool_id=hid)
        always_gen_succ += int(gen.success)

        hint = hid if pg >= 0.5 else None
        ad = run_episode(task=t, rng=rng, hint_tool_id=hint)
        adaptive_succ += int(ad.success)

    n = len(tasks)
    return {
        "n": n,
        "base_sr": base_succ / n,
        "always_generate_sr": always_gen_succ / n,
        "adaptive_sr": adaptive_succ / n,
        "avg_generate_prob": float(p_generate.mean().item()),
    }
