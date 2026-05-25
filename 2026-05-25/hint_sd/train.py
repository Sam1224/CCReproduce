from __future__ import annotations

import argparse
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from env import GridTask, MiniGridWorld
from hindsight import select_failure_relevant_steps
from model import PolicyNet


def set_seed(seed: int) -> np.random.Generator:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    return rng


@torch.no_grad()
def sample_action(logits: torch.Tensor) -> int:
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def run_epoch(
    *,
    env: MiniGridWorld,
    rng: np.random.Generator,
    student: PolicyNet,
    teacher: PolicyNet,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mode: str,
    batch_tasks: int,
    horizon: int,
    max_hindsight_steps: int,
) -> Dict[str, float]:
    student.train()
    teacher.eval()

    supervised_steps = 0
    total_steps = 0
    successes = 0

    losses: List[float] = []

    for _ in range(batch_tasks):
        task = env.sample_task(rng, horizon=horizon)

        def policy_fn(state_np: np.ndarray) -> int:
            state = torch.from_numpy(state_np).to(device).unsqueeze(0)
            logits = student(state).logits.squeeze(0)
            return sample_action(logits)

        states_np, actions, reward = env.rollout(task, policy_fn)
        total_steps += len(actions)
        successes += int(reward == 1)

        if reward == 1:
            continue

        if mode == "dense":
            # per-turn feedback baseline: supervise every step
            selected = []
            pos = task.start
            for t, act in enumerate(actions):
                fb = env.optimal_action(pos, task.goal)
                selected.append((t, fb))
                pos = env.step(pos, act)
        elif mode == "hint_sd":
            signals = select_failure_relevant_steps(
                env, task, states_np, actions, reward, max_steps=max_hindsight_steps
            )
            selected = [(s.step_index, s.feedback_action) for s in signals]
        else:
            raise ValueError(f"unknown mode: {mode}")

        if not selected:
            continue

        # build a tiny supervised batch over selected steps
        step_states = torch.from_numpy(np.stack([states_np[t] for t, _ in selected], axis=0)).to(device)
        feedback_actions = torch.tensor([fb for _, fb in selected], dtype=torch.long, device=device)

        # teacher (privileged) distribution
        with torch.no_grad():
            teacher_logits = teacher(step_states, feedback_action=feedback_actions).logits
            teacher_probs = torch.softmax(teacher_logits, dim=-1)

        student_logits = student(step_states).logits
        student_log_probs = torch.log_softmax(student_logits, dim=-1)

        # KL(teacher || student)
        loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))
        supervised_steps += len(selected)

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "success_rate": successes / max(1, batch_tasks),
        "supervised_steps": supervised_steps,
        "total_steps": total_steps,
    }


@torch.no_grad()
def evaluate(env: MiniGridWorld, rng: np.random.Generator, student: PolicyNet, device: torch.device, n: int, horizon: int) -> float:
    student.eval()
    success = 0
    for _ in range(n):
        task = env.sample_task(rng, horizon=horizon)

        def policy_fn(state_np: np.ndarray) -> int:
            state = torch.from_numpy(state_np).to(device).unsqueeze(0)
            logits = student(state).logits.squeeze(0)
            action = int(torch.argmax(logits).item())
            return action

        _, _, reward = env.rollout(task, policy_fn)
        success += int(reward == 1)
    return success / max(1, n)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["hint_sd", "dense"], default="hint_sd")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_tasks", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--max_hindsight_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save", type=str, default="checkpoints/hint_sd_student.pt")
    args = parser.parse_args()

    device = torch.device(args.device)
    rng = set_seed(args.seed)

    env = MiniGridWorld(size=5)
    student = PolicyNet(use_feedback=False).to(device)
    teacher = PolicyNet(use_feedback=True).to(device)

    # Pretrain the teacher as a feedback-conditioned oracle: it should output the feedback_action.
    # This is a toy stand-in for the paper's feedback-conditioned teacher distribution.
    teacher_opt = torch.optim.AdamW(teacher.parameters(), lr=1e-3)
    teacher.train()
    for _ in range(200):
        states = torch.from_numpy(rng.random((128, 5), dtype=np.float32)).to(device)
        fb = torch.tensor(rng.integers(0, 4, size=(128,)), dtype=torch.long, device=device)
        logits = teacher(states, feedback_action=fb).logits
        loss = F.cross_entropy(logits, fb)
        teacher_opt.zero_grad(set_to_none=True)
        loss.backward()
        teacher_opt.step()
    teacher.eval()

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr)

    best = 0.0
    start = time.time()
    for epoch in tqdm(range(1, args.epochs + 1), desc=f"train({args.mode})"):
        metrics = run_epoch(
            env=env,
            rng=rng,
            student=student,
            teacher=teacher,
            optimizer=optimizer,
            device=device,
            mode=args.mode,
            batch_tasks=args.batch_tasks,
            horizon=args.horizon,
            max_hindsight_steps=args.max_hindsight_steps,
        )
        val = evaluate(env, rng, student, device, n=400, horizon=args.horizon)
        best = max(best, val)

        if epoch % 10 == 0 or epoch == 1:
            steps_ratio = metrics["supervised_steps"] / max(1, metrics["total_steps"])
            tqdm.write(
                f"epoch={epoch:03d} loss={metrics['loss']:.4f} train_succ={metrics['success_rate']:.3f} "
                f"val_succ={val:.3f} supervised/total={steps_ratio:.3f}"
            )

    elapsed = time.time() - start
    print(f"best_val_success={best:.3f} elapsed_sec={elapsed:.1f}")

    ckpt_path = args.save
    import os

    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save({"state_dict": student.state_dict(), "mode": args.mode, "horizon": args.horizon}, ckpt_path)
    print(f"saved={ckpt_path}")


if __name__ == "__main__":
    main()
