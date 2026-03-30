from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from dataset import make_dataset, split
from model import CoTVideoModel, onehot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["supervised", "rl"], required=True)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-3)
    args = ap.parse_args()

    tr, te = split(make_dataset(n=8000, seed=1), 0.85)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CoTVideoModel().to(device)

    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / "cot_video.pt"
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device)["state_dict"], strict=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for ep in range(args.epochs):
        model.train()

        if args.mode == "supervised":
            step_logits = model.step_logits(tr.video.to(device))
            step_loss = torch.nn.functional.cross_entropy(step_logits, tr.step.to(device))
            ans_logits = model(tr.video.to(device), onehot(tr.step.to(device), 5))
            ans_loss = torch.nn.functional.cross_entropy(ans_logits, tr.y.to(device))
            loss = step_loss + ans_loss
        else:
            # REINFORCE on sampled step; reward is 1 if answer correct.
            step_logits = model.step_logits(tr.video.to(device))
            dist = torch.distributions.Categorical(logits=step_logits)
            sampled = dist.sample()
            ans_logits = model(tr.video.to(device), onehot(sampled, 5))
            pred = ans_logits.argmax(dim=-1)
            reward = (pred == tr.y.to(device)).float().detach()
            # Baseline for variance reduction.
            b = reward.mean()
            loss = -((reward - b) * dist.log_prob(sampled)).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            step_pred = model.step_logits(te.video.to(device)).argmax(dim=-1)
            ans = model(te.video.to(device), onehot(step_pred, 5)).argmax(dim=-1).cpu()
        acc = (ans == te.y).float().mean().item()
        print(f"epoch={ep} mode={args.mode} loss={loss.item():.4f} acc={acc:.3f}")

    torch.save({"state_dict": model.state_dict()}, ckpt_path)


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
