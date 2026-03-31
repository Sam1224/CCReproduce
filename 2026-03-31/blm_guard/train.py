from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from dataset import make_dataset, split
from model import BLMGuardModel, aks_bin_top, group_advantage, hybrid_reward, logprob_from_logits, select_regions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--rl_steps", type=int, default=300)
    ap.add_argument("--group", type=int, default=6)
    args = ap.parse_args()

    data = make_dataset(n=8000, seed=0)
    tr, _ = split(data, 0.85)

    model = BLMGuardModel()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Supervised warmup
    for ep in range(args.epochs):
        model.train()
        total = 0.0
        n = 0
        for ex in tr[:1400]:
            # risk-prompt selection proxy: use frame L2 norm as risk score
            scores = ex.frames.norm(dim=-1)
            idx = aks_bin_top(scores, m=3)
            key = ex.frames[idx]
            region = select_regions(key).mean(dim=0)

            s_logits, t_logits, r_logit = model(region, ex.asr)

            loss = (
                F.cross_entropy(s_logits.unsqueeze(0), torch.tensor([ex.label_scene]))
                + F.cross_entropy(t_logits.unsqueeze(0), torch.tensor([ex.label_type]))
                + F.binary_cross_entropy_with_logits(r_logit.unsqueeze(0), torch.tensor([float(ex.label_risky)]))
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total += float(loss.detach())
            n += 1

        print(f"sup epoch={ep} loss={total/max(1,n):.4f}")

    # RL: GRPO-style update on (scene,type) actions
    model.train()
    for step in range(args.rl_steps):
        ex = tr[step % len(tr)]
        scores = ex.frames.norm(dim=-1)
        idx = aks_bin_top(scores, m=3)
        region = select_regions(ex.frames[idx]).mean(dim=0)

        with torch.no_grad():
            s_logits, t_logits, _ = model(region, ex.asr)
            s_logits = s_logits.unsqueeze(0).repeat(args.group, 1)
            t_logits = t_logits.unsqueeze(0).repeat(args.group, 1)

            s_act = torch.multinomial(torch.softmax(s_logits, dim=-1), num_samples=1).squeeze(1)
            t_act = torch.multinomial(torch.softmax(t_logits, dim=-1), num_samples=1).squeeze(1)

            rewards = torch.tensor(
                [
                    hybrid_reward(int(s_act[i]), int(t_act[i]), ex.label_scene, ex.label_type)
                    for i in range(args.group)
                ],
                dtype=torch.float32,
            )

        adv = group_advantage(rewards)

        s_logits, t_logits, _ = model(region, ex.asr)
        s_logits = s_logits.unsqueeze(0).repeat(args.group, 1)
        t_logits = t_logits.unsqueeze(0).repeat(args.group, 1)

        lp = logprob_from_logits(s_logits, s_act) + logprob_from_logits(t_logits, t_act)
        loss = -(adv.detach() * lp).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 50 == 0:
            print(f"rl step={step} reward_mean={rewards.mean().item():.3f} loss={loss.item():.4f}")

    ckpt = Path("checkpoints")
    ckpt.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, ckpt / "blm_guard.pt")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
