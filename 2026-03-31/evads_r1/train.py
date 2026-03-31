from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from dataset import make_dataset, split
from model import EVAdsPolicy, mg_grpo_advantage, multinomial_logprob


def reward_fn(action: int, label: int, n_classes: int) -> float:
    # Strict: exact match
    if action == label:
        return 1.0

    # Relaxed: same coarse bucket (multi-grained reward)
    # (mimics "relaxed" grading in the paper)
    bucket_a = action // max(1, n_classes // 3)
    bucket_y = label // max(1, n_classes // 3)
    return 0.5 if bucket_a == bucket_y else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--rl_steps", type=int, default=400)
    ap.add_argument("--group", type=int, default=6)
    args = ap.parse_args()

    data = make_dataset(n=6000, seed=0)
    tr, _ = split(data, 0.85)

    model = EVAdsPolicy(vocab=1000, d_v=48, d=128, n_classes=12)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Supervised warmup
    for ep in range(args.epochs):
        model.train()
        total = 0.0
        n = 0
        for ex in tr[:1200]:
            logits = model(ex.video, ex.asr, ex.ocr, ex.q).unsqueeze(0)
            loss = F.cross_entropy(logits, torch.tensor([ex.y]))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach())
            n += 1
        print(f"sup epoch={ep} loss={total/max(1,n):.4f}")

    # RL specialization (simplified MG-GRPO)
    model.train()
    for step in range(args.rl_steps):
        ex = tr[step % len(tr)]

        # sample a group of answers
        with torch.no_grad():
            logits = model(ex.video, ex.asr, ex.ocr, ex.q).unsqueeze(0).repeat(args.group, 1)
            probs = torch.softmax(logits, dim=-1)
            actions = torch.multinomial(probs, num_samples=1).squeeze(1)
            rewards = torch.tensor([reward_fn(int(a), ex.y, 12) for a in actions], dtype=torch.float32)

        adv = mg_grpo_advantage(rewards)

        # re-run forward for gradients
        logits = model(ex.video, ex.asr, ex.ocr, ex.q).unsqueeze(0).repeat(args.group, 1)
        logp = multinomial_logprob(logits, actions)
        # maximize reward => minimize -adv*logp
        loss = -(adv.detach() * logp).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 50 == 0:
            print(f"rl step={step} reward_mean={rewards.mean().item():.3f} loss={loss.item():.4f}")

    ckpt = Path("checkpoints")
    ckpt.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, ckpt / "evads_r1.pt")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
