from __future__ import annotations

import argparse

import torch
from tqdm import tqdm

from data import build_world, sample_batch
from model import GRPolicy
from utils import load_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--num_eval", type=int, default=1000)
    p.add_argument("--beam", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = f"{args.ckpt_dir}/ckpt.pt"
    config, state = load_checkpoint(ckpt_path, device)

    world = build_world(
        num_items=int(config["num_items"]),
        token_vocab=int(config["token_vocab"]),
        semantic_len=int(config["semantic_len"]),
        emb_dim=32,
        seed=int(config["seed"]),
    )

    policy = GRPolicy(
        num_items=int(config["num_items"]),
        token_vocab=int(config["token_vocab"]),
        semantic_len=int(config["semantic_len"]),
        emb_dim=int(config["emb_dim"]),
        hidden_dim=int(config["hidden_dim"]),
    ).to(device)
    policy.load_state_dict(state)
    policy.eval()

    hits = 0
    total = 0
    halluc = 0
    total_cands = 0

    batch_size = 64
    steps = (args.num_eval + batch_size - 1) // batch_size

    for step in tqdm(range(steps), desc="Eval"):
        bsz = min(batch_size, args.num_eval - step * batch_size)
        batch = sample_batch(
            world,
            batch_size=bsz,
            history_len=int(config["history_len"]),
            seed=args.seed + step,
        )

        hist = batch.history_item_ids.to(device)
        tgt_item = batch.target_item_id

        beams, _ = policy.beam_search(hist, beam_size=args.beam)
        # beams: [B, beam, L]

        for i in range(bsz):
            ok = False
            for j in range(beams.shape[1]):
                seq = tuple(int(x) for x in beams[i, j].tolist())
                item = world.seq_to_item.get(seq, -1)
                if item < 0:
                    halluc += 1
                total_cands += 1
                if item == int(tgt_item[i]):
                    ok = True
            hits += int(ok)
            total += 1

    hr10 = hits / max(total, 1)
    halluc_rate = halluc / max(total_cands, 1)

    print(f"HR@{args.beam}: {hr10:.4f}")
    print(f"Hallucination rate (invalid semantic id in top-{args.beam}): {halluc_rate:.4f}")
    print(f"Mode: {config.get('mode')}")


if __name__ == "__main__":
    main()
