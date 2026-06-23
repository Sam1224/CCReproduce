"""
train_listwise.py — Stage 2: List-wise Feedback Alignment (paper Sec. 3.4.2).

List-wise Softmax DPO over a behavior-ranked preference list (6 levels,
Table 2) with monotonic-CTR filtering (done in data.py). A frozen reference
policy (the Stage-1 checkpoint) provides l_ref for the log-ratios r_i (Eq. 6).

Example (smoke test):
    python train_listwise.py --tiny --max_steps 3 --save_dir outputs/listwise
"""

from __future__ import annotations

import argparse
import copy
import os

import torch
from torch.utils.data import DataLoader

from data import generate_dataset, ListwiseDataset, listwise_collate
from model import OneBarGenerator
from losses import listwise_dpo_loss


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="facebook/bart-base")
    ap.add_argument("--init_from", default="",
                    help="Stage-1 checkpoint dir (else fresh / tiny model)")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--n_train", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--lambda_sft", type=float, default=1.0)
    ap.add_argument("--prompt_style", default="compressed")
    ap.add_argument("--save_dir", default="outputs/listwise")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def build_model(args):
    name = args.init_from if args.init_from else args.model_name
    return OneBarGenerator(name, tiny=args.tiny, offline=args.offline)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    policy = build_model(args)
    tok = policy.tokenizer

    # frozen reference = snapshot of the policy at Stage-2 start (Eq. 6).
    ref = copy.deepcopy(policy)
    for p in ref.parameters():
        p.requires_grad_(False)
    ref.eval()

    samples = generate_dataset(n=args.n_train, seed=args.seed)
    ds = ListwiseDataset(samples, tok, prompt_style=args.prompt_style)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    collate_fn=listwise_collate)

    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    policy.train()

    step = 0
    for epoch in range(args.epochs):
        for batch in dl:                       # batch = list of samples
            opt.zero_grad()
            batch_loss = torch.zeros(())
            n_used = 0
            for s in batch:
                n_cand = s["cand_input_ids"].size(0)
                # expand the shared source to one row per candidate
                src_ids = s["input_ids"].unsqueeze(0).expand(n_cand, -1)
                src_mask = s["attention_mask"].unsqueeze(0).expand(n_cand, -1)
                cand_labels = s["cand_labels"]      # [n_cand, T] (-100 pads)

                pol_logps = policy.sequence_logprob(src_ids, src_mask, cand_labels)
                with torch.no_grad():
                    ref_logps = ref.sequence_logprob(src_ids, src_mask, cand_labels)

                loss = listwise_dpo_loss(
                    pol_logps, ref_logps, s["levels"],
                    beta=args.beta, temperature=args.temperature,
                    lambda_sft=args.lambda_sft)
                batch_loss = batch_loss + loss
                n_used += 1

            batch_loss = batch_loss / max(n_used, 1)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            step += 1
            print(f"[Listwise-DPO] epoch {epoch} step {step} "
                  f"loss {batch_loss.item():.4f}")
            if args.max_steps > 0 and step >= args.max_steps:
                break
        if args.max_steps > 0 and step >= args.max_steps:
            break

    os.makedirs(args.save_dir, exist_ok=True)
    policy.model.save_pretrained(args.save_dir)
    tok.save_pretrained(args.save_dir)
    print(f"[Listwise-DPO] saved checkpoint -> {args.save_dir}")


if __name__ == "__main__":
    main()
