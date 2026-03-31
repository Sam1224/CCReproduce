from __future__ import annotations

import argparse
import copy
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import SyntheticSIDDataset, collate, parse_prediction
from model import CausalTransformerLM, lm_loss, token_logprob


def reward_fn(pred_sid: int, ok_format: bool, target_sid: int) -> float:
    r = 0.0
    r += 1.0 if pred_sid == int(target_sid) else 0.0
    r += 0.1 if ok_format else 0.0
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=str, choices=["sft", "rl"], required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--rollouts", type=int, default=16)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--kl_beta", type=float, default=1e-3)
    ap.add_argument("--ckpt", type=str, default="checkpoints/sid_reasoner.pt")
    args = ap.parse_args()

    ds = SyntheticSIDDataset(seed=args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=collate)

    model = CausalTransformerLM(vocab=ds.vocab.vocab_size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    if os.path.exists(args.ckpt):
        payload = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(payload["model"], strict=False)

    pad_id = 0
    eos_id = ds.vocab.stoi["<eos>"]

    if args.stage == "sft":
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []
            for batch in loader:
                ids = batch["ids"].to(device)
                logits = model(ids, pad_id=pad_id)
                loss = lm_loss(logits, ids, pad_id=pad_id)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                losses.append(float(loss.detach().cpu()))

            print(f"sft epoch={epoch} loss={np.mean(losses):.4f}")

    else:
        # GRPO-style RL on outcome reward.
        ref = copy.deepcopy(model).eval()
        for p in ref.parameters():
            p.requires_grad = False

        it = iter(loader)
        for step in range(1, args.steps + 1):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)

            ids = batch["ids"].to(device)
            prompt_len = batch["prompt_len"].to(device)
            target_sid = batch["target_sid"].to(device)

            B = int(ids.shape[0])
            # prompt is ids up to prompt_len
            max_pl = int(prompt_len.max().item())
            prompt = ids[:, :max_pl]

            K = int(args.rollouts)
            rewards = torch.zeros(K, B, device=device)
            seqs = []
            logp_ref = []

            with torch.no_grad():
                for k in range(K):
                    gen, _ = model.generate(prompt=prompt, max_new_tokens=8, eos_id=eos_id)
                    seqs.append(gen)

                    ref_logits = ref(gen)
                    lp = token_logprob(ref_logits, gen).sum(dim=-1)
                    logp_ref.append(lp)

                    for i in range(B):
                        pred, ok = parse_prediction(gen[i], ds.vocab)
                        rewards[k, i] = reward_fn(pred, ok, int(target_sid[i]))

            adv = (rewards - rewards.mean(dim=0, keepdim=True)) / (rewards.std(dim=0, keepdim=True) + 1e-6)

            policy = 0.0
            kl = 0.0
            for k in range(K):
                gen = seqs[k]
                cur_logits = model(gen)
                lp = token_logprob(cur_logits, gen).sum(dim=-1)
                policy = policy + (-adv[k] * lp).mean()
                kl = kl + (lp - logp_ref[k]).mean()

            policy = policy / K
            kl = kl / K

            loss = policy + float(args.kl_beta) * kl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step % 50 == 0:
                print(f"rl step={step} loss={float(loss.detach().cpu()):.4f} reward={float(rewards.mean().detach().cpu()):.3f}")

    os.makedirs(os.path.dirname(args.ckpt), exist_ok=True)
    torch.save({"model": model.state_dict(), "args": vars(args)}, args.ckpt)
    print(f"saved: {args.ckpt}")


if __name__ == "__main__":
    main()
