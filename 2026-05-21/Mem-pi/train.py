import argparse
import os

import torch

from mem_pi import (
    MemPiPolicy,
    TrainConfig,
    build_offline_experience_bank,
    build_vocab,
    evaluate,
    generate_tasks,
    train_stage1_experience_distillation,
    train_stage2_adaptation_distillation,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="checkpoints")
    args = parser.parse_args()

    cfg = TrainConfig(seed=args.seed, device=args.device)

    num_tools = 4
    vocab = build_vocab(num_tools)

    train_tasks = generate_tasks(n=5000, num_tools=num_tools, explicit_ratio=0.45, seed=args.seed)
    eval_tasks = generate_tasks(n=1200, num_tools=num_tools, explicit_ratio=0.45, seed=args.seed + 1)

    experience_bank = build_offline_experience_bank(train_tasks)

    policy = MemPiPolicy(vocab_size=len(vocab.token_to_id), num_tools=num_tools)

    policy = train_stage1_experience_distillation(policy=policy, vocab=vocab, experience_bank=experience_bank, cfg=cfg)

    os.makedirs(args.out, exist_ok=True)
    stage1_path = f"{args.out}/stage1.pt"
    torch.save({"state_dict": policy.state_dict(), "vocab": vocab.token_to_id}, stage1_path)

    policy = train_stage2_adaptation_distillation(policy=policy, vocab=vocab, tasks=train_tasks, cfg=cfg)

    stage2_path = f"{args.out}/stage2.pt"
    torch.save({"state_dict": policy.state_dict(), "vocab": vocab.token_to_id}, stage2_path)

    metrics = evaluate(policy=policy, vocab=vocab, tasks=eval_tasks, seed=args.seed)
    print("[EVAL]", metrics)
    print(f"Saved checkpoints to: {args.out}")


if __name__ == "__main__":
    main()
