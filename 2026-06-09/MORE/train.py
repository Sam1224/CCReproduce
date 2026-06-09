from __future__ import annotations

import argparse

import numpy as np
import torch
from tqdm import tqdm

from data import Sample, build_dataset, build_vocab, decode, encode
from model import Seq2SeqGRU
from rewards import AdaptiveWeights, bleu2, correctness_reward, length_reward, unique_trigram_ratio
from utils import ensure_dir, save_checkpoint, save_json, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--mode", type=str, choices=["fixed", "more"], default="more")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--emb_dim", type=int, default=96)
    p.add_argument("--hidden_dim", type=int, default=192)

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--sft_epochs", type=int, default=8)
    p.add_argument("--rl_steps", type=int, default=800)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--temperature", type=float, default=0.9)

    p.add_argument("--train_size", type=int, default=2000)
    p.add_argument("--test_size", type=int, default=500)
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

    train_samples = build_dataset(n=args.train_size, seed=args.seed)
    test_samples = build_dataset(n=args.test_size, seed=args.seed + 1)

    vocab = build_vocab(train_samples + test_samples)
    inv_vocab = {v: k for k, v in vocab.items()}

    pad_id = vocab["<pad>"]
    bos_id = vocab["<bos>"]
    eos_id = vocab["<eos>"]

    model = Seq2SeqGRU(vocab_size=len(vocab), emb_dim=args.emb_dim, hidden_dim=args.hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    ensure_dir(args.out_dir)

    metrics = {"mode": args.mode, "sft": [], "rl": []}

    # ---------------- SFT ----------------
    model.train()
    for epoch in range(args.sft_epochs):
        rng = np.random.default_rng(args.seed + epoch)
        idx = rng.permutation(len(train_samples)).tolist()

        total_loss = 0.0
        total_n = 0
        for start in tqdm(range(0, len(idx), args.batch_size), desc=f"SFT ep{epoch+1}"):
            batch = [train_samples[i] for i in idx[start : start + args.batch_size]]

            src_ids = [encode(s.input_text, vocab, add_bos=False, add_eos=True) for s in batch]
            tgt_ids = [encode(s.target_text, vocab, add_bos=True, add_eos=True) for s in batch]

            src, src_len = pad_batch(src_ids, pad_id)
            tgt, _ = pad_batch(tgt_ids, pad_id)

            src = src.to(device)
            src_len = src_len.to(device)
            tgt = tgt.to(device)

            loss = model.nll(src, src_len, tgt)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total_loss += float(loss.item()) * len(batch)
            total_n += len(batch)

        metrics["sft"].append({"epoch": epoch + 1, "nll": total_loss / max(total_n, 1)})

    # ---------------- RL fine-tuning ----------------
    aw = AdaptiveWeights()

    for step in tqdm(range(args.rl_steps), desc=args.mode.upper()):
        rng = np.random.default_rng(args.seed + 10_000 + step)
        batch = [train_samples[int(i)] for i in rng.integers(0, len(train_samples), size=(args.batch_size,))]

        src_ids = [encode(s.input_text, vocab, add_bos=False, add_eos=True) for s in batch]
        tgt_ids = [encode(s.target_text, vocab, add_bos=True, add_eos=True) for s in batch]

        src, src_len = pad_batch(src_ids, pad_id)
        tgt, _ = pad_batch(tgt_ids, pad_id)

        src = src.to(device)
        src_len = src_len.to(device)

        # sample from current policy
        seq, _ = model.generate(src, src_len, bos_id=bos_id, eos_id=eos_id, temperature=args.temperature)
        lp = model.sequence_logprob(src, src_len, seq)

        rewards = []
        bleu_rs = []
        len_rs = []
        cor_rs = []
        tri_rs = []

        for i, s in enumerate(batch):
            gen_text = decode(seq[i].tolist(), inv_vocab)
            ref_text = s.target_text

            cor = correctness_reward(credit_limit=s.credit_limit, eligible=s.eligible, text=gen_text)
            b = bleu2(ref_text, gen_text)
            l = length_reward(gen_text)
            tri = unique_trigram_ratio(gen_text)

            if args.mode == "fixed":
                r = 0.7 * cor + 0.3 * b
            else:
                # MORE-style: correctness as constraint / scaffold
                if cor < 1.0:
                    r = -1.0
                else:
                    aw.update(bleu_r=b, len_r=l)
                    r = 1.0 + aw.w_bleu * b + aw.w_len * l + 0.2 * tri

            rewards.append(r)
            bleu_rs.append(b)
            len_rs.append(l)
            cor_rs.append(cor)
            tri_rs.append(tri)

        r_t = torch.tensor(rewards, dtype=torch.float32, device=device)
        loss = -(r_t.detach() * lp).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if (step + 1) % 100 == 0:
            metrics["rl"].append(
                {
                    "step": step + 1,
                    "loss": float(loss.item()),
                    "reward": float(np.mean(rewards)),
                    "correct": float(np.mean(cor_rs)),
                    "bleu2": float(np.mean(bleu_rs)),
                    "len_r": float(np.mean(len_rs)),
                    "tri": float(np.mean(tri_rs)),
                    "w_bleu": float(aw.w_bleu),
                    "w_len": float(aw.w_len),
                }
            )

    ckpt_path = f"{args.out_dir}/ckpt.pt"
    config = {
        **vars(args),
        "vocab_size": len(vocab),
    }
    save_checkpoint(ckpt_path, config=config, model=model, vocab=vocab)
    save_json(f"{args.out_dir}/train_metrics.json", metrics)

    print(f"[OK] saved checkpoint to: {ckpt_path}")


if __name__ == "__main__":
    main()
