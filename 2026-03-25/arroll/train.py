from __future__ import annotations

import numpy as np
import torch

from policy import DigitPolicy, PolicyConfig
from rollout import rollout_pruned_sampling, rollout_sampling
from toy_task import sample_episode


def reinforce_update(
    policy: DigitPolicy,
    rollouts,
    baseline: float,
    lr: float,
):
    opt = torch.optim.AdamW(policy.parameters(), lr=lr)

    losses = []
    for r in rollouts:
        adv = r.reward - baseline
        losses.append(-adv * r.logp)

    loss = sum(losses) / max(1, len(losses))
    opt.zero_grad()
    loss.backward()
    opt.step()

    return float(loss.item())


def run(mode: str, seed: int = 0) -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rng = np.random.default_rng(seed)

    policy = DigitPolicy(PolicyConfig(hidden=64)).to(device)

    steps = 120
    length = 6
    n_rollouts = 32

    baseline = 0.0
    beta = 0.9

    tokens_total = 0
    rewards_total = 0.0

    for step in range(steps):
        ep = sample_episode(rng, length=length, target_max=30)

        if mode == 'prune':
            rollouts, tokens = rollout_pruned_sampling(policy, ep.target, ep.length, n_rollouts, device, seed + step)
        else:
            rollouts, tokens = rollout_sampling(policy, ep.target, ep.length, n_rollouts, device, seed + step)

        tokens_total += tokens
        avg_r = sum(r.reward for r in rollouts) / len(rollouts)
        rewards_total += avg_r

        baseline = beta * baseline + (1 - beta) * avg_r

        # Make logp differentiable by re-running the policy on each sequence.
        # This keeps the toy demo simple (not efficient).
        # For real RLVR you’d cache states/logprobs during rollout.
        policy.train()
        opt = torch.optim.AdamW(policy.parameters(), lr=2e-3)
        loss = 0.0

        for r in rollouts:
            # rebuild logp under current policy
            tgt = torch.tensor([[float(ep.target)]], device=device)
            toks = torch.tensor([r.tokens], device=device, dtype=torch.long)
            # compute per-step logits
            lp = torch.zeros((1,), device=device)
            for t in range(1, toks.shape[1]):
                logits = policy(tgt, toks[:, :t])
                logprob = torch.log_softmax(logits, dim=-1)
                lp = lp + logprob.gather(1, toks[:, t : t + 1]).squeeze(1)

            adv = float(r.reward - baseline)
            loss = loss + (-adv) * lp.mean()

        loss = loss / len(rollouts)
        opt.zero_grad()
        loss.backward()
        opt.step()
        policy.eval()

        if (step + 1) % 30 == 0:
            avg_reward = rewards_total / (step + 1)
            tok_per_step = tokens_total / (step + 1)
            print(
                f'[{mode}] step={step+1} avg_reward={avg_reward:.3f} '
                f'tokens/step={tok_per_step:.1f} baseline={baseline:.3f}'
            )

    avg_reward = rewards_total / steps
    tok_per_step = tokens_total / steps
    print(f'\nFinal [{mode}] avg_reward={avg_reward:.3f} tokens/step={tok_per_step:.1f}')


def main() -> None:
    print('Training without pruning...')
    run('base', seed=0)

    print('\nTraining with online pruning...')
    run('prune', seed=123)


if __name__ == '__main__':
    main()
