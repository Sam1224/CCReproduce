from __future__ import annotations

import argparse

import numpy as np
import torch

from env import MiniGridWorld
from model import PolicyNet


@torch.no_grad()
def evaluate(student: PolicyNet, device: torch.device, n: int, horizon: int) -> float:
    rng = np.random.default_rng(123)
    env = MiniGridWorld(size=5)
    student.eval()

    success = 0
    for _ in range(n):
        task = env.sample_task(rng, horizon=horizon)

        def policy_fn(state_np: np.ndarray) -> int:
            state = torch.from_numpy(state_np).to(device).unsqueeze(0)
            logits = student(state).logits.squeeze(0)
            return int(torch.argmax(logits).item())

        _, _, reward = env.rollout(task, policy_fn)
        success += int(reward == 1)

    return success / max(1, n)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--episodes", type=int, default=2000)
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device)

    student = PolicyNet(use_feedback=False).to(device)
    student.load_state_dict(ckpt["state_dict"])

    horizon = int(ckpt.get("horizon", 12))
    sr = evaluate(student, device, n=args.episodes, horizon=horizon)
    print(f"success_rate={sr:.4f} episodes={args.episodes} horizon={horizon}")


if __name__ == "__main__":
    main()
