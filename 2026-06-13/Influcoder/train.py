import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from data import build_toy_dataset
from model import (
    CountSketch,
    ToyConfig,
    ToyDecoderClassifier,
    ToyEncoder,
    auprc,
    cosine_sim,
    per_example_lora_grad_sketch,
    spearmanr,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="runs/influcoder")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--decoder_steps", type=int, default=800)
    parser.add_argument("--encoder_steps", type=int, default=1200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    cfg = ToyConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = build_toy_dataset(cfg, seed=args.seed)
    vocab_size = len(ds["vocab"]["id_to_token"])

    pool_x = torch.tensor(ds["pool"]["x"], dtype=torch.long, device=device)
    pool_m = torch.tensor(ds["pool"]["m"], dtype=torch.long, device=device)
    pool_y = torch.tensor(ds["pool"]["y"], dtype=torch.long, device=device)

    query_x = torch.tensor(ds["query"]["x"], dtype=torch.long, device=device)
    query_m = torch.tensor(ds["query"]["m"], dtype=torch.long, device=device)

    # ----------------
    # Stage-1: train a tiny decoder classifier (with LoRA on the final layer)
    # ----------------
    dec = ToyDecoderClassifier(
        vocab_size=vocab_size,
        d_model=cfg.d_model,
        pad_id=int(ds["vocab"]["pad_id"]),
        lora_r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
    ).to(device)

    opt = torch.optim.AdamW(dec.parameters(), lr=args.lr, weight_decay=1e-2)
    dec.train()

    rng = np.random.RandomState(args.seed)
    for step in range(1, args.decoder_steps + 1):
        idx = torch.tensor(rng.randint(0, pool_x.size(0), size=(args.batch_size,)), device=device)
        logits = dec(pool_x[idx], pool_m[idx])
        loss = F.cross_entropy(logits, pool_y[idx])

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0)
        opt.step()

        if step % 200 == 0:
            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                acc = float((pred == pool_y[idx]).float().mean().item())
            print(f"[InflucoderToy][decoder] step={step:04d} loss={loss.item():.4f} acc={acc:.3f}")

    # ----------------
    # Stage-1: compute ground-truth influence embeddings via sketched LoRA gradients
    # ----------------
    dec.eval()
    for p in dec.parameters():
        p.requires_grad_(True)

    # create CountSketch after observing LoRA grad dimensionality
    with torch.no_grad():
        dummy_logits = dec(pool_x[:1], pool_m[:1])
        _ = dummy_logits.sum()

    # LoRA grad dim
    lora_dim = sum(p.numel() for p in dec.lora_parameters())
    sketch = CountSketch(in_dim=lora_dim, out_dim=cfg.sketch_dim, seed=args.seed)

    def compute_sketches(x: torch.Tensor, m: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = []
        for i in range(x.size(0)):
            out.append(per_example_lora_grad_sketch(dec, x[i], m[i], y[i], sketch=sketch))
        return torch.stack(out, dim=0)

    # queries: we need labels to backprop; use label=1 ("toxic") to mimic tracing toxic behavior sources
    query_y = torch.ones(query_x.size(0), dtype=torch.long, device=device)

    print("[InflucoderToy] computing ground-truth sketches…")
    pool_sk = compute_sketches(pool_x, pool_m, pool_y)
    query_sk = compute_sketches(query_x, query_m, query_y)

    # ----------------
    # Stage-2: distill to an encoder that predicts influence embeddings
    # ----------------
    enc = ToyEncoder(
        vocab_size=vocab_size,
        d_model=cfg.d_model,
        pad_id=int(ds["vocab"]["pad_id"]),
        out_dim=cfg.enc_dim,
    ).to(device)
    opt2 = torch.optim.AdamW(enc.parameters(), lr=args.lr, weight_decay=1e-2)

    enc.train()
    for step in range(1, args.encoder_steps + 1):
        q_idx = torch.tensor(rng.randint(0, query_x.size(0), size=(args.batch_size,)), device=device)
        p_idx = torch.tensor(rng.randint(0, pool_x.size(0), size=(args.batch_size,)), device=device)

        zq = enc(query_x[q_idx], query_m[q_idx])
        zp = enc(pool_x[p_idx], pool_m[p_idx])
        pred_sim = cosine_sim(zq, zp)

        with torch.no_grad():
            tgt_sim = cosine_sim(query_sk[q_idx], pool_sk[p_idx])

        loss = F.mse_loss(pred_sim, tgt_sim)

        opt2.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
        opt2.step()

        if step % 300 == 0:
            print(f"[InflucoderToy][encoder] step={step:04d} loss={loss.item():.6f}")

    # ----------------
    # quick eval on training set (toy)
    # ----------------
    enc.eval()

    with torch.no_grad():
        zq_all = enc(query_x, query_m)
        zp_all = enc(pool_x, pool_m)

        # evaluate ranking correlation averaged over queries
        corr = []
        for qi in range(query_x.size(0)):
            tgt = cosine_sim(query_sk[qi : qi + 1], pool_sk).squeeze(0)
            pred = cosine_sim(zq_all[qi : qi + 1], zp_all).squeeze(0)
            corr.append(spearmanr(pred, tgt))
        mean_spearman = float(np.mean(corr))

        # evaluate toxicity filtering AUPRC using a fixed query (q0)
        scores = cosine_sim(zq_all[0:1], zp_all).squeeze(0)
        ap = auprc(pool_y, scores)

    metrics = {
        "mean_spearman": mean_spearman,
        "auprc_toxic_filtering": float(ap),
        "n_pool": int(pool_x.size(0)),
        "n_query": int(query_x.size(0)),
        "sketch_dim": int(cfg.sketch_dim),
        "enc_dim": int(cfg.enc_dim),
    }

    ckpt = {
        "cfg": ds["cfg"],
        "vocab": ds["vocab"],
        "decoder": dec.state_dict(),
        "encoder": enc.state_dict(),
        "countsketch": {"in_dim": int(lora_dim), "out_dim": int(cfg.sketch_dim), "h": sketch.h.tolist(), "s": sketch.s.tolist()},
        "metrics": metrics,
    }

    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
    torch.save(ckpt, ckpt_path)

    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\n=== Metrics (toy) ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print(f"\nsaved: {ckpt_path}")


if __name__ == "__main__":
    main()
