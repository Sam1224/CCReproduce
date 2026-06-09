from __future__ import annotations

import argparse

import numpy as np
import torch
from tqdm import tqdm

from data import build_dataset, decode, encode
from model import Seq2SeqGRU
from rewards import bleu2, correctness_reward, unique_trigram_ratio
from utils import load_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", type=str, required=True)
    p.add_argument("--num_eval", type=int, default=500)
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def pad_batch(seqs: list[list[int]], pad_id: int) -> tuple[torch.LongTensor, torch.LongTensor]:
    lens = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    max_len = int(lens.max().item())
    out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return out, lens


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = f"{args.ckpt_dir}/ckpt.pt"
    config, state, vocab = load_checkpoint(ckpt_path, device)
    inv_vocab = {v: k for k, v in vocab.items()}

    pad_id = vocab["<pad>"]
    bos_id = vocab["<bos>"]
    eos_id = vocab["<eos>"]

    model = Seq2SeqGRU(vocab_size=len(vocab), emb_dim=int(config["emb_dim"]), hidden_dim=int(config["hidden_dim"]))
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    base_n = int(config.get("test_size", 500))
    test_samples = build_dataset(n=max(args.num_eval, base_n), seed=int(config["seed"]) + 1)[: args.num_eval]

    cors = []
    bleus = []
    tris = []

    bs = 64
    for start in tqdm(range(0, len(test_samples), bs), desc="Eval"):
        batch = test_samples[start : start + bs]

        src_ids = [encode(s.input_text, vocab, add_bos=False, add_eos=True) for s in batch]
        src, src_len = pad_batch(src_ids, pad_id)
        src = src.to(device)
        src_len = src_len.to(device)

        seq = model.greedy(src, src_len, bos_id=bos_id, eos_id=eos_id)

        for i, s in enumerate(batch):
            gen_text = decode(seq[i].tolist(), inv_vocab)
            cors.append(correctness_reward(credit_limit=s.credit_limit, eligible=s.eligible, text=gen_text))
            bleus.append(bleu2(s.target_text, gen_text))
            tris.append(unique_trigram_ratio(gen_text))

    print(f"Mode: {config.get('mode')}")
    print(f"Correctness acc: {float(np.mean(cors)):.4f}")
    print(f"BLEU-2: {float(np.mean(bleus)):.4f}")
    print(f"Unique trigram ratio: {float(np.mean(tris)):.4f}")


if __name__ == "__main__":
    main()
