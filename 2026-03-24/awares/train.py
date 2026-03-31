from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from dataset import AwaResToyDataset, batch_iter_sft, get_crop_lowres, get_crop_lowres_torch
from model import AwaResConfig, AwaResModel


def eval_acc(model: AwaResModel, ds: AwaResToyDataset, *, n: int = 200) -> tuple[float, float]:
    model.eval()
    correct = 0
    used_crop = 0
    with torch.no_grad():
        for i in range(min(n, len(ds))):
            ex = ds[i]
            img_low = torch.from_numpy(ex.img_low).unsqueeze(0).unsqueeze(0)
            action_logits, ans_low, low_feat = model.forward_low(img_low)
            action = int(action_logits.argmax(dim=-1).item())
            if action == 0:
                pred = int(ans_low.argmax(dim=-1).item())
            else:
                used_crop += 1
                crop_idx = action - 1
                crop = torch.from_numpy(get_crop_lowres(ex.img_high, crop_id=crop_idx)).unsqueeze(0).unsqueeze(0)
                ans = model.forward_with_crop(low_feat, crop)
                pred = int(ans.argmax(dim=-1).item())
            correct += int(pred == ex.label)
    model.train()
    return correct / max(1, min(n, len(ds))), used_crop / max(1, min(n, len(ds)))


def sft_train(model: AwaResModel, *, train_ds: AwaResToyDataset, val_ds: AwaResToyDataset, steps: int) -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    it = batch_iter_sft(train_ds, batch_size=64, seed=0)

    for step in range(1, steps + 1):
        b = next(it)
        action_logits, ans_low, low_feat = model.forward_low(b.img_low)
        ans_fused = model.forward_with_crop(low_feat, b.img_crop)

        loss_action = F.cross_entropy(action_logits, b.action)

        no_crop = b.action == 0
        loss_answer = torch.zeros((), dtype=torch.float32)
        if no_crop.any():
            loss_answer = loss_answer + F.cross_entropy(ans_low[no_crop], b.label[no_crop])
        if (~no_crop).any():
            loss_answer = loss_answer + F.cross_entropy(ans_fused[~no_crop], b.label[~no_crop])

        loss = loss_action + loss_answer

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 100 == 0:
            acc, crop_rate = eval_acc(model, val_ds, n=300)
            print(f"[SFT] step={step} loss={loss.item():.4f} val_acc={acc:.3f} crop_rate={crop_rate:.3f}")


def rl_finetune(model: AwaResModel, *, train_ds: AwaResToyDataset, val_ds: AwaResToyDataset, steps: int) -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    it = batch_iter_sft(train_ds, batch_size=64, seed=1)

    baseline = 0.0
    beta = 0.05
    crop_cost = 0.15

    for step in range(1, steps + 1):
        b = next(it)

        action_logits, ans_low, low_feat = model.forward_low(b.img_low)
        dist = torch.distributions.Categorical(logits=action_logits)
        actions = dist.sample()

        rewards = torch.zeros(actions.shape[0])

        pred = ans_low.argmax(dim=-1)
        use_crop = actions != 0
        used_crop_count = int(use_crop.sum().item())
        if use_crop.any():
            crop_ids = actions[use_crop] - 1
            crop_batch = get_crop_lowres_torch(b.img_high[use_crop], crop_id=crop_ids)
            ans = model.forward_with_crop(low_feat[use_crop], crop_batch)
            pred[use_crop] = ans.argmax(dim=-1)

        correct = (pred == b.label).float()
        rewards = correct - crop_cost * use_crop.float()

        baseline = (1 - beta) * baseline + beta * float(rewards.mean().item())
        adv = rewards - baseline

        loss_pg = -(dist.log_prob(actions) * adv.detach()).mean()
        loss_ans = F.cross_entropy(ans_low, b.label)
        loss = loss_pg + 0.2 * loss_ans

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 100 == 0:
            acc, crop_rate = eval_acc(model, val_ds, n=300)
            print(
                f"[RL] step={step} loss={loss.item():.4f} mean_r={rewards.mean().item():.3f} used_crop={used_crop_count} val_acc={acc:.3f} crop_rate={crop_rate:.3f}"
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-steps", type=int, default=500)
    ap.add_argument("--rl-steps", type=int, default=500)
    ap.add_argument("--out", default="checkpoints/awares.pt")
    args = ap.parse_args()

    train_ds = AwaResToyDataset(n_samples=6000, seed=0)
    val_ds = AwaResToyDataset(n_samples=1000, seed=1)

    model = AwaResModel(AwaResConfig())

    sft_train(model, train_ds=train_ds, val_ds=val_ds, steps=args.sft_steps)
    rl_finetune(model, train_ds=train_ds, val_ds=val_ds, steps=args.rl_steps)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)

    acc, crop_rate = eval_acc(model, val_ds, n=500)
    print(f"done: val_acc={acc:.3f}, crop_rate={crop_rate:.3f}, saved={out}")


if __name__ == "__main__":
    main()
