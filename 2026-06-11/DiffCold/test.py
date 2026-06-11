import argparse
import os

import numpy as np
import torch

from data import ToyRecConfig, recall_ndcg_at_k
from model import ContentProjector, DiffColdDDPM, DiffusionConfig


def _rank(user_emb: np.ndarray, item_emb: np.ndarray) -> np.ndarray:
    scores = user_emb @ item_emb.T
    return np.argsort(-scores, axis=1)


def _report(name: str, ranks: np.ndarray, test_warm: np.ndarray, test_cold: np.ndarray, warm_ids: set[int], cold_ids: set[int]):
    # overall
    overall_pos = np.concatenate([test_warm, test_cold], axis=1)
    r_o, n_o = recall_ndcg_at_k(ranks, overall_pos, k=20)

    # warm-only
    r_w, n_w = recall_ndcg_at_k(ranks, test_warm, k=20)

    # cold-only
    r_c, n_c = recall_ndcg_at_k(ranks, test_cold, k=20)

    print(
        f"{name:10s} | Overall R@20={r_o:.4f} N@20={n_o:.4f} | Cold R@20={r_c:.4f} N@20={n_c:.4f} | Warm R@20={r_w:.4f} N@20={n_w:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default="runs/diffcold")
    args = parser.parse_args()

    ckpt_path = os.path.join(args.ckpt_dir, "ckpt.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    toy_cfg = ToyRecConfig(**ckpt["toy_cfg"])
    dcfg = DiffusionConfig(**ckpt["diffusion_cfg"])

    data = ckpt["data"]

    content = torch.tensor(data["content"], dtype=torch.float32)
    item_emb_true = torch.tensor(data["item_emb_true"], dtype=torch.float32)
    user_emb = np.asarray(data["user_emb"], dtype=np.float32)

    warm_ids = np.asarray(data["warm_ids"], dtype=np.int64)
    cold_ids = np.asarray(data["cold_ids"], dtype=np.int64)

    test_warm = np.asarray(data["test_warm"], dtype=np.int64)
    test_cold = np.asarray(data["test_cold"], dtype=np.int64)

    start = torch.tensor(data["start"], dtype=torch.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    proj = ContentProjector(toy_cfg.content_dim, toy_cfg.embed_dim)
    proj.load_state_dict(ckpt["proj"])
    proj.to(device).eval()

    ddpm = DiffColdDDPM(dcfg)
    ddpm.load_state_dict(ckpt["ddpm"])
    ddpm.to(device).eval()

    content = content.to(device)
    item_emb_true = item_emb_true.to(device)
    start = start.to(device)

    # Backbone: warm is known, cold unknown -> zeros
    item_backbone = torch.zeros_like(item_emb_true)
    item_backbone[warm_ids] = item_emb_true[warm_ids]
    ranks = _rank(user_emb, item_backbone.cpu().numpy())
    _report("Backbone", ranks, test_warm, test_cold, set(warm_ids.tolist()), set(cold_ids.tolist()))

    # ContentProj: use projected content for all items (often seesaw)
    item_proj = proj(content).detach().cpu().numpy()
    ranks = _rank(user_emb, item_proj)
    _report("ContentProj", ranks, test_warm, test_cold, set(warm_ids.tolist()), set(cold_ids.tolist()))

    # DiffCold: warm keep backbone, cold via diffusion
    item_diff = item_backbone.clone()
    with torch.no_grad():
        cold_emb = ddpm.sample(content=content[cold_ids], start=start[cold_ids])
    item_diff[cold_ids] = cold_emb

    ranks = _rank(user_emb, item_diff.cpu().numpy())
    _report("DiffCold", ranks, test_warm, test_cold, set(warm_ids.tolist()), set(cold_ids.tolist()))


if __name__ == "__main__":
    main()
