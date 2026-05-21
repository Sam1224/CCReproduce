import argparse

import torch

from mem_pi import MemPiPolicy, Vocab, build_vocab, evaluate, generate_tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/stage2.pt")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    num_tools = 4
    vocab = build_vocab(num_tools)

    ckpt = torch.load(args.ckpt, map_location=args.device)
    policy = MemPiPolicy(vocab_size=len(vocab.token_to_id), num_tools=num_tools)
    policy.load_state_dict(ckpt["state_dict"])
    policy.to(args.device)

    eval_tasks = generate_tasks(n=1200, num_tools=num_tools, explicit_ratio=0.45, seed=123)
    metrics = evaluate(policy=policy, vocab=vocab, tasks=eval_tasks, seed=123)
    print("[EVAL]", metrics)


if __name__ == "__main__":
    main()
