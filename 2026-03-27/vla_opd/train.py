from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from env import GridEnv, build_offline_dataset
from model import Policy


def sft_train(model: Policy, data, epochs: int, lr: float) -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    X = torch.stack([t.obs for t in data], dim=0)
    y = torch.tensor([t.act for t in data], dtype=torch.long)

    for ep in range(epochs):
        model.train()
        logits = model(X)
        loss = torch.nn.functional.cross_entropy(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        acc = (logits.argmax(dim=-1) == y).float().mean().item()
        print(f"sft epoch={ep} loss={loss.item():.4f} acc={acc:.3f}")


def rl_train(model: Policy, episodes: int, lr: float, gamma: float = 0.99) -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    env = GridEnv(size=5)

    for ep in range(episodes):
        obs = env.reset()
        logps = []
        rewards = []

        for _ in range(20):
            logits = model(obs.unsqueeze(0))[0]
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            logps.append(dist.log_prob(a))
            obs, r, done = env.step(int(a.item()))
            rewards.append(r)
            if done:
                break

        # returns
        G = 0.0
        rets = []
        for r in reversed(rewards):
            G = r + gamma * G
            rets.append(G)
        rets = list(reversed(rets))
        rets_t = torch.tensor(rets, dtype=torch.float32)
        rets_t = (rets_t - rets_t.mean()) / (rets_t.std().clamp_min(1e-6))

        loss = -(torch.stack(logps) * rets_t).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if (ep + 1) % 100 == 0:
            print(f"rl ep={ep+1} steps={len(rewards)} return={sum(rewards):.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["sft", "rl"], required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--lr", type=float, default=3e-3)
    args = ap.parse_args()

    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / "policy.pt"

    model = Policy()
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu")["state_dict"], strict=True)

    if args.stage == "sft":
        data = build_offline_dataset(n=2500, seed=1)
        sft_train(model, data, epochs=args.epochs, lr=args.lr)
    else:
        rl_train(model, episodes=args.episodes, lr=1e-3)

    torch.save({"state_dict": model.state_dict()}, ckpt_path)


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
