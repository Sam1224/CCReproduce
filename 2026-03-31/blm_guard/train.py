from __future__ import annotations

import argparse
import copy
import os
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import (
    BLMGuardDataset,
    SPECIAL_TOKENS,
    collate,
    consistency_reward,
    correctness_reward,
    format_reward,
)
from model import BLMGuardModel, cross_entropy_lm, logprob_from_logits


def classification_loss(out, scene_gt, type_gt, risky_gt) -> torch.Tensor:
    return (
        F.cross_entropy(out.scene_logits, scene_gt)
        + F.cross_entropy(out.type_logits, type_gt)
        + F.cross_entropy(out.risky_logits, risky_gt)
    )


def accuracy_score(y_true, y_pred) -> float:
    y_true = list(y_true)
    y_pred = list(y_pred)
    if not y_true:
        return 0.0
    hit = sum(int(a == b) for a, b in zip(y_true, y_pred))
    return hit / len(y_true)


@torch.no_grad()
def evaluate(model: BLMGuardModel, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    ys = []
    ps = []
    yt = []
    pt = []
    yr = []
    pr = []

    for batch in loader:
        for k in batch:
            batch[k] = batch[k].to(device)

        out = model(
            frames=batch["frames"],
            patches=batch["patches"],
            frame_mask=batch["frame_mask"],
            asr_ids=batch["asr_ids"],
            ocr_ids=batch["ocr_ids"],
            prompt_ids=batch["prompt_ids"],
        )

        ys.extend(batch["scene"].cpu().tolist())
        ps.extend(out.scene_logits.argmax(dim=-1).cpu().tolist())

        yt.extend(batch["ad_type"].cpu().tolist())
        pt.extend(out.type_logits.argmax(dim=-1).cpu().tolist())

        yr.extend(batch["risky"].cpu().tolist())
        pr.extend(out.risky_logits.argmax(dim=-1).cpu().tolist())

    return {
        "scene_acc": float(accuracy_score(ys, ps)),
        "type_acc": float(accuracy_score(yt, pt)),
        "risky_acc": float(accuracy_score(yr, pr)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=7)

    ap.add_argument("--rl_steps", type=int, default=200)
    ap.add_argument("--rollouts", type=int, default=8)
    ap.add_argument("--kl_beta", type=float, default=1e-3)

    ap.add_argument("--ckpt", type=str, default="checkpoints/blm_guard.pt")
    args = ap.parse_args()

    ds = BLMGuardDataset(jsonl_path=args.data or None, seed=args.seed)
    n = len(ds)
    split = int(n * 0.9)
    train_ds = torch.utils.data.Subset(ds, list(range(split)))
    val_ds = torch.utils.data.Subset(ds, list(range(split, n)))

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, collate_fn=collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab_size = len(ds.vocab.stoi)
    model = BLMGuardModel(vocab=vocab_size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    pad_id = SPECIAL_TOKENS["<pad>"]
    bos_id = SPECIAL_TOKENS["<bos>"]
    eos_id = SPECIAL_TOKENS["<eos>"]

    # ------------------ supervised warmup ------------------
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            for k in batch:
                batch[k] = batch[k].to(device)

            out = model(
                frames=batch["frames"],
                patches=batch["patches"],
                frame_mask=batch["frame_mask"],
                asr_ids=batch["asr_ids"],
                ocr_ids=batch["ocr_ids"],
                prompt_ids=batch["prompt_ids"],
            )

            cls = classification_loss(out, batch["scene"], batch["ad_type"], batch["risky"])
            lm_logits = model.decoder(batch["rationale_ids"], out.memory, memory_key_padding_mask=out.memory_pad_mask)
            lm = cross_entropy_lm(lm_logits, batch["rationale_ids"], pad_id=pad_id)

            loss = cls + 0.5 * lm
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))

        metrics = evaluate(model, val_loader, device)
        print(f"sup epoch={epoch} loss={np.mean(losses):.4f} metrics={metrics}")

    # ------------------ GRPO-style RL fine-tune ------------------
    ref = copy.deepcopy(model).eval()
    for p in ref.parameters():
        p.requires_grad = False

    model.train()
    it = iter(train_loader)
    for step in range(1, args.rl_steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)

        for k in batch:
            batch[k] = batch[k].to(device)

        out = model(
            frames=batch["frames"],
            patches=batch["patches"],
            frame_mask=batch["frame_mask"],
            asr_ids=batch["asr_ids"],
            ocr_ids=batch["ocr_ids"],
            prompt_ids=batch["prompt_ids"],
        )

        # Rollout K sequences per example.
        K = int(args.rollouts)
        rewards = torch.zeros(K, batch["scene"].shape[0], device=device)
        seqs = []
        logp_ref = []

        with torch.no_grad():
            # reference logprobs for KL
            for k_ in range(K):
                ids, _ = model.decoder.generate(
                    memory=out.memory,
                    memory_key_padding_mask=out.memory_pad_mask,
                    bos_id=bos_id,
                    eos_id=eos_id,
                    max_len=batch["rationale_ids"].shape[1],
                )
                seqs.append(ids)

                ref_logits = ref.decoder(ids, out.memory, memory_key_padding_mask=out.memory_pad_mask)
                lp = logprob_from_logits(ref_logits, ids).sum(dim=-1)
                logp_ref.append(lp)

                sp = out.scene_logits.argmax(dim=-1)
                tp = out.type_logits.argmax(dim=-1)
                rp = out.risky_logits.argmax(dim=-1)

                for i in range(ids.shape[0]):
                    r = 0.0
                    r += correctness_reward(int(sp[i]), int(tp[i]), int(rp[i]), int(batch["scene"][i]), int(batch["ad_type"][i]), int(batch["risky"][i]))
                    r += 0.2 * format_reward(ids[i])
                    r += 0.3 * consistency_reward(ids[i], int(sp[i]), int(tp[i]), int(rp[i]), ds.vocab)
                    rewards[k_, i] = r

        # Group-wise advantage normalization.
        adv = (rewards - rewards.mean(dim=0, keepdim=True)) / (rewards.std(dim=0, keepdim=True) + 1e-6)

        policy_loss = 0.0
        kl = 0.0
        for k_ in range(K):
            ids = seqs[k_]
            cur_logits = model.decoder(ids, out.memory, memory_key_padding_mask=out.memory_pad_mask)
            lp = logprob_from_logits(cur_logits, ids).sum(dim=-1)

            policy_loss = policy_loss + (-adv[k_] * lp).mean()
            kl = kl + (lp - logp_ref[k_]).mean()

        policy_loss = policy_loss / K
        kl = kl / K

        # Keep classification head aligned.
        cls = classification_loss(out, batch["scene"], batch["ad_type"], batch["risky"])

        loss = cls + policy_loss + float(args.kl_beta) * kl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 50 == 0:
            model.eval()
            metrics = evaluate(model, val_loader, device)
            model.train()
            print(f"rl step={step} loss={float(loss.detach().cpu()):.4f} metrics={metrics}")

    os.makedirs(os.path.dirname(args.ckpt), exist_ok=True)
    torch.save({"model": model.state_dict(), "vocab": ds.vocab.stoi, "args": vars(args)}, args.ckpt)
    print(f"saved: {args.ckpt}")


if __name__ == "__main__":
    main()
