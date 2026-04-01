from __future__ import annotations

import argparse
import json
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-users", type=int, default=200)
    parser.add_argument("--num-items", type=int, default=1000)
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    sequences = []
    for _ in range(args.num_users):
        seq = [rng.randrange(args.num_items) for _ in range(args.seq_len)]
        sequences.append(seq)

    obj = {"num_items": args.num_items, "sequences": sequences}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(obj, f)


if __name__ == "__main__":
    main()
