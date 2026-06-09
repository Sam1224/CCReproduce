from __future__ import annotations

import argparse
import time

import torch
from tqdm import tqdm

from data import build_world, sample_batch
from model import GRPolicy
from reward import ToyRewardModel, reward_discriminability
from utils import ensure_dir, save_checkpoint, save_json, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--mode", type=str, choices=["sft", "grpo", "adagrpo"], default="adagrpo")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--num_items", type=int, default=200)
    p.add_argument("--token_vocab", type=int, default=64)
    p.add_argument("--semantic_len", type=int, default=3)
    p.add_argument("--emb_dim", type=int, default=64)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--history_len", type=int, default=20)

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--sft_steps", type=int, default=800)
    p.add_argument("--rl_steps", type=int, default=800)
    p.add_argument("--lr", type=float, default=3e-4)

    p.add_argument("--k", type=int, default=16, help="#candidates per sample for GRPO")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--lambda_grpo", type=float, default=0.2)

    p.add_argument("--difficulty_rank_frac", type=float, default=0.6)
    p.add_argument("--rm_margin", type=float, default=0.2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    world = build_world(
        num_items=args.num_items,
        token_vocab=args.token_vocab,
        semantic_len=args.semantic_len,
        emb_dim=32,
        seed=args.seed,
    )

    policy = GRPolicy(
        num_items=args.num_items,
        token_vocab=args.token_vocab,
        semantic_len=args.semantic_len,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)

    rm = ToyRewardModel(world)

    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)

    metrics = {
        "mode": args.mode,
        "seed": args.seed,
        "device": str(device),
        "sft": [],
        "rl": [],
    }

    ensure_dir(args.out_dir)

    # ---------------- SFT ----------------
    policy.train()
    for step in tqdm(range(args.sft_steps), desc="SFT"):
        batch = sample_batch(world, batch_size=args.batch_size, history_len=args.history_len, seed=args.seed + step)
        hist = batch.history_item_ids.to(device)
        tgt = batch.target_semantic_ids.to(device)

        loss = policy.nll(hist, tgt)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if (step + 1) % 100 == 0:
            metrics["sft"].append({"step": step + 1, "nll": float(loss.item())})

    # ---------------- RL fine-tuning ----------------
    if args.mode in ("grpo", "adagrpo"):
        for step in tqdm(range(args.rl_steps), desc=args.mode.upper()):
            batch = sample_batch(
                world,
                batch_size=args.batch_size,
                history_len=args.history_len,
                seed=args.seed + 10_000 + step,
            )
            hist = batch.history_item_ids.to(device)
            tgt_item = batch.target_item_id.to(device)
            tgt_seq = batch.target_semantic_ids.to(device)

            nll = policy.nll(hist, tgt_seq)

            cand_seq, cand_lp_sample = policy.sample_k(hist, k=args.k, temperature=args.temperature)
            # cand_seq: [B,K,L]
            # We recompute logprob with teacher forcing for stability
            cand_lp = policy.sequence_logprob_k(hist, cand_seq)  # [B,K]

            rm_out = rm.score(target_item_id=tgt_item.cpu(), cand_seq=cand_seq.cpu())
            reward = rm_out.reward.to(device)

            adv = (reward - reward.mean(dim=1, keepdim=True)) / (reward.std(dim=1, keepdim=True) + 1e-6)

            # per-sample GRPO loss
            loss_grpo_per = -(adv.detach() * cand_lp).mean(dim=1)  # [B]

            if args.mode == "grpo":
                clip = torch.ones_like(loss_grpo_per)
            else:
                # policy-side difficulty: GT logprob ranks low among sampled candidates
                gt_lp = policy.sequence_logprob(hist, tgt_seq)  # [B]
                worse = (cand_lp > gt_lp.unsqueeze(1)).sum(dim=1).float()
                rank_frac = worse / float(args.k)
                difficult = rank_frac >= args.difficulty_rank_frac

                # reward discriminability: GT is present and beats negatives by margin
                eq = (cand_seq == tgt_seq.unsqueeze(1)).all(dim=-1)  # [B,K]
                gt_reward = torch.where(eq, reward, torch.full_like(reward, float("-inf"))).max(dim=1).values
                disc = reward_discriminability(
                    reward=reward,
                    target_in_candidates=eq,
                    gt_reward=gt_reward,
                    margin=args.rm_margin,
                )

                clip = (difficult & disc).float()

            loss = nll + args.lambda_grpo * (clip * loss_grpo_per).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if (step + 1) % 100 == 0:
                metrics["rl"].append(
                    {
                        "step": step + 1,
                        "nll": float(nll.item()),
                        "grpo": float(loss_grpo_per.mean().item()),
                        "clip_rate": float(clip.mean().item()),
                        "reward_mean": float(reward.mean().item()),
                    }
                )

    ckpt_path = f"{args.out_dir}/ckpt.pt"
    config = vars(args)
    save_checkpoint(ckpt_path, config=config, model=policy)
    save_json(f"{args.out_dir}/train_metrics.json", metrics)

    print(f"[OK] saved checkpoint to: {ckpt_path}")


if __name__ == "__main__":
    main()
