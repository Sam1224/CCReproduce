from __future__ import annotations

import os
from pathlib import Path

import torch

from env import TASK_TYPES, make_env
from model import Router


def main() -> None:
    ckpt = torch.load("checkpoints/router.pt", map_location="cpu")
    env = make_env(seed=1, agents=5)
    router = Router(num_tasks=len(TASK_TYPES), num_agents=env.agent_skill.shape[0])
    router.load_state_dict(ckpt["state_dict"], strict=True)
    router.eval()

    # Show learned routing preference.
    probs = torch.softmax(router.logits, dim=-1)
    for i, t in enumerate(TASK_TYPES):
        p = probs[i].tolist()
        best = int(torch.argmax(probs[i]).item())
        print(f"task={t} probs={['%.2f'%x for x in p]} best_agent={best}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
