"""
ClaimDiff-RL training script.

Paper training algorithm (Section 3.3):
  Uses GRPO (Group Relative Policy Optimization) with ClaimDiff rewards.
  For each training batch:
    1. Sample G candidate captions from current policy π_θ per image.
    2. Run ClaimDiff Judge on each candidate → per-difference statistics.
    3. Compute ClaimDiff rewards for each candidate.
    4. Normalize rewards within the group (GRPO advantage estimation).
    5. Update policy via policy gradient with group-normalized advantages.

  The key innovation is that the reward R = -α·R_hall - β·R_omit is decomposed,
  allowing fine-grained control over the hallucination-omission tradeoff.
"""
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import ToyCapDataset, collate_fn
from judge import ClaimDiffJudge
from model import ToyCaptionPolicy, text_to_image_feat, encode_text, decode, TOY_VOCAB
from reward import RewardConfig, compute_reward


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--group_size", type=int, default=4,
                   help="G: number of candidate captions sampled per image (GRPO)")
    p.add_argument("--alpha", type=float, default=1.0, help="Hallucination penalty weight")
    p.add_argument("--beta", type=float, default=1.0, help="Omission penalty weight")
    p.add_argument("--temperature", type=float, default=1.2,
                   help="Sampling temperature for caption generation")
    p.add_argument("--save", type=str, default="checkpoints/claimdiff_rl.pt")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def grpo_loss(
    log_probs: torch.Tensor,  # (G, T-1)
    advantages: torch.Tensor,  # (G,)
    eps_clip: float = 0.2,
) -> torch.Tensor:
    """
    GRPO policy gradient loss (simplified PPO-style clipping).

    Paper equation (adapted from GRPO):
      L = -E[ min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t) ]
    where r_t = π_θ(a_t|s_t) / π_old(a_t|s_t)

    In this simplified implementation we use REINFORCE with group-normalized advantages
    (equivalent to GRPO when π_old ≈ π_θ at the start of each rollout).

      L = -E[log π_θ(a|s) * A]

    The key insight: advantages are computed within the group → no external value baseline needed.
    """
    # Sequence-level log prob: sum over time steps, mean over group
    seq_log_prob = log_probs.sum(dim=-1)  # (G,)
    loss = -(seq_log_prob * advantages).mean()
    return loss


def normalize_advantages(rewards: list) -> torch.Tensor:
    """
    GRPO group normalization: A_i = (R_i - mean(R)) / (std(R) + ε)
    This removes the need for a learned value baseline.
    """
    r = torch.tensor(rewards, dtype=torch.float32)
    return (r - r.mean()) / (r.std() + 1e-8)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    dataset = ToyCapDataset()
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    model = ToyCaptionPolicy().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    judge = ClaimDiffJudge()
    reward_cfg = RewardConfig(alpha=args.alpha, beta=args.beta)

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        epoch_reward = 0.0
        n_batches = 0

        for batch in loader:
            image_desc = batch["image_descs"][0]
            reference = batch["references"][0]

            # Compute image feature (toy: from text description)
            img_feat = text_to_image_feat(image_desc, d_model=128, device=device).to(device)
            img_feat = img_feat.expand(args.group_size, -1)  # (G, D)

            # ── Step 1: Sample G candidates from current policy ─────────────────
            generated_ids = model.generate(img_feat, max_new_tokens=20,
                                           temperature=args.temperature)  # (G, T)

            candidates = []
            for g in range(args.group_size):
                ids = generated_ids[g].tolist()
                text = decode(ids) or "unknown"
                candidates.append(text)

            # ── Step 2: Run ClaimDiff Judge on each candidate ───────────────────
            rewards_raw = []
            for cand in candidates:
                diffs = judge.judge(cand, reference, image_desc)
                r = compute_reward(diffs, cand, reference, reward_cfg)
                rewards_raw.append(r["R_total"])

            # ── Step 3: GRPO advantage normalization ────────────────────────────
            advantages = normalize_advantages(rewards_raw).to(device)  # (G,)

            # ── Step 4: Compute policy gradient loss ────────────────────────────
            # Need log probs of the generated sequences under the current policy.
            # Pad sequences to the same length.
            max_len = generated_ids.size(1)
            log_probs_list = []
            for g in range(args.group_size):
                ids = generated_ids[g].unsqueeze(0)  # (1, T)
                # Prepend BOS for teacher-forcing input
                bos = torch.full((1, 1), ToyCaptionPolicy.BOS_ID, dtype=torch.long, device=device)
                full_ids = torch.cat([bos, ids], dim=1)  # (1, T+1)
                lp = model.log_prob_of(full_ids, img_feat[g:g+1])  # (1, T)
                log_probs_list.append(lp.squeeze(0))  # (T,)

            # Pad to equal length
            padded = torch.nn.utils.rnn.pad_sequence(
                log_probs_list, batch_first=True, padding_value=0.0
            )  # (G, T_max)

            loss = grpo_loss(padded, advantages)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_reward += sum(rewards_raw) / args.group_size
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_reward = epoch_reward / max(n_batches, 1)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | loss={avg_loss:.4f} | avg_reward={avg_reward:.4f}")

    torch.save(model.state_dict(), args.save)
    print(f"\nModel saved to {args.save}")


if __name__ == "__main__":
    main()
