from __future__ import annotations

import os
from pathlib import Path

import torch

from env import GridEnv
from model import Policy


def run(policy: Policy, episodes: int = 200) -> float:
    env = GridEnv(size=5)
    success = 0

    for _ in range(episodes):
        obs = env.reset()
        for _ in range(20):
            with torch.no_grad():
                a = int(policy(obs.unsqueeze(0))[0].argmax().item())
            obs, _, done = env.step(a)
            if done:
                success += 1
                break

    return success / episodes


def main() -> None:
    ckpt = torch.load("checkpoints/policy.pt", map_location="cpu")
    policy = Policy()
    policy.load_state_dict(ckpt["state_dict"], strict=True)
    policy.eval()

    rate = run(policy, 200)
    print(f"success_rate={rate:.3f}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
