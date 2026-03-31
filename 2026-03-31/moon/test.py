from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn.functional as F

from dataset import make_pairs, make_synthetic_catalog, split
from model import MoonEncoder


def recall_at_k(sim: torch.Tensor, ks=(1, 5, 10)) -> dict:
    # sim: (N, N)
    gt = torch.arange(sim.shape[0])
    out = {}
    for k in ks:
        topk = sim.topk(k=k, dim=-1).indices
        hit = (topk == gt.unsqueeze(-1)).any(dim=-1).float().mean().item()
        out[k] = hit
    return out


def main() -> None:
    products, meta = make_synthetic_catalog(n_products=2500, seed=2)
    pairs = make_pairs(products, n_pairs=1200, seed=3)
    _, te = split(pairs, 0.0)  # all test

    ckpt_path = Path("checkpoints/moon.pt")
    model = MoonEncoder(vocab_size=meta["vocab_size"], d_model=128, d_img=meta["d_img"])
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state["state_dict"], strict=False)

    model.eval()

    q = []
    k = []
    for ex in te:
        q.append(model.encode_text(ex.query_token_ids, ex.query_token_types))
        p = ex.product
        k.append(model.encode_product(p.token_ids, p.token_types, p.images_full, p.images_core))

    q = F.normalize(torch.stack(q, dim=0), dim=-1)
    k = F.normalize(torch.stack(k, dim=0), dim=-1)
    sim = q @ k.t()

    r = recall_at_k(sim)
    print("Recall@K:")
    for kk, vv in r.items():
        print(f"  R@{kk}: {vv:.3f}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
