from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from env import TASK_TYPES, make_env, step
from model import Router


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=0.1)
    args = ap.parse_args()

    rng = torch.Generator().manual_seed(1)
    env = make_env(seed=1, agents=5)

    router = Router(num_tasks=len(TASK_TYPES), num_agents=env.agent_skill.shape[0])
    opt = torch.optim.SGD(router.parameters(), lr=args.lr)

    rewards = []

    for ep in range(args.episodes):
        task = int(torch.randint(0, len(TASK_TYPES), (1,), generator=rng).item())
        logits = router(torch.tensor([task]))[0]
        dist = torch.distributions.Categorical(logits=logits)
        agent = int(dist.sample().item())

        ok, r = step(env, task, agent, rng)
        rewards.append(r)

        loss = -torch.tensor(r, dtype=torch.float32) * dist.log_prob(torch.tensor(agent))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if (ep + 1) % 200 == 0:
            avg = sum(rewards[-200:]) / 200
            print(f"ep={ep+1} avg_reward={avg:.3f}")

    ckpt = Path("checkpoints")
    ckpt.mkdir(exist_ok=True)
    torch.save({"state_dict": router.state_dict(), "agent_skill": env.agent_skill}, ckpt / "router.pt")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
