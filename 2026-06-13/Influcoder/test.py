import argparse
import json
import os

import numpy as np
import torch

from data import build_toy_dataset
from model import CountSketch, ToyConfig, ToyDecoderClassifier, ToyEncoder, auprc, cosine_sim, per_example_lora_grad_sketch, spearmanr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default="runs/influcoder")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    ckpt_path = os.path.join(args.ckpt_dir, "ckpt.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    cfg = ToyConfig()
    ds = build_toy_dataset(cfg, seed=args.seed)
    vocab_size = len(ds["vocab"]["id_to_token"])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dec = ToyDecoderClassifier(
        vocab_size=vocab_size,
        d_model=cfg.d_model,
        pad_id=int(ds["vocab"]["pad_id"]),
        lora_r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
    ).to(device)
    dec.load_state_dict(ckpt["decoder"], strict=False)
    dec.eval()

    enc = ToyEncoder(
        vocab_size=vocab_size,
        d_model=cfg.d_model,
        pad_id=int(ds["vocab"]["pad_id"]),
        out_dim=cfg.enc_dim,
    ).to(device)
    enc.load_state_dict(ckpt["encoder"], strict=False)
    enc.eval()

    pool_x = torch.tensor(ds["pool"]["x"], dtype=torch.long, device=device)
    pool_m = torch.tensor(ds["pool"]["m"], dtype=torch.long, device=device)
    pool_y = torch.tensor(ds["pool"]["y"], dtype=torch.long, device=device)

    query_x = torch.tensor(ds["query"]["x"], dtype=torch.long, device=device)
    query_m = torch.tensor(ds["query"]["m"], dtype=torch.long, device=device)
    query_y = torch.ones(query_x.size(0), dtype=torch.long, device=device)

    # rebuild CountSketch
    cs = ckpt["countsketch"]
    sketch = CountSketch(in_dim=int(cs["in_dim"]), out_dim=int(cs["out_dim"]), seed=0)
    sketch.h = torch.tensor(cs["h"], dtype=torch.long)
    sketch.s = torch.tensor(cs["s"], dtype=torch.long)

    def compute_sketches(x: torch.Tensor, m: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = []
        for i in range(x.size(0)):
            out.append(per_example_lora_grad_sketch(dec, x[i], m[i], y[i], sketch=sketch))
        return torch.stack(out, dim=0)

    print("[InflucoderToy] computing sketches for eval…")
    pool_sk = compute_sketches(pool_x, pool_m, pool_y)
    query_sk = compute_sketches(query_x, query_m, query_y)

    with torch.no_grad():
        zq_all = enc(query_x, query_m)
        zp_all = enc(pool_x, pool_m)

        corr = []
        for qi in range(query_x.size(0)):
            tgt = cosine_sim(query_sk[qi : qi + 1], pool_sk).squeeze(0)
            pred = cosine_sim(zq_all[qi : qi + 1], zp_all).squeeze(0)
            corr.append(spearmanr(pred, tgt))
        mean_spearman = float(np.mean(corr))

        scores = cosine_sim(zq_all[0:1], zp_all).squeeze(0)
        ap = auprc(pool_y, scores)

    metrics = {
        "mean_spearman": mean_spearman,
        "auprc_toxic_filtering": float(ap),
        "n_pool": int(pool_x.size(0)),
        "n_query": int(query_x.size(0)),
    }

    print("\n=== Metrics (toy, eval on new seed) ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    with open(os.path.join(args.ckpt_dir, "metrics_eval.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
