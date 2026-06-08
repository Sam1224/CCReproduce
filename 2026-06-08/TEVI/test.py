from __future__ import annotations

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from dataset import ToyConfig, ToyMultimodalDataset
from metrics import cosine_sim, recall_at_k
from models import IdentitySparseAutoencoder, SAEConfig, TEVI, TEVIMasker, TopKSparseAutoencoder


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg_path = os.path.join(args.ckpt_dir, "config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg_data = json.load(f)

    toy_cfg = ToyConfig(**cfg_data["toy"])
    # keep model dims aligned with training args
    embed_dim = int(toy_cfg.embed_dim)

    ckpt = torch.load(os.path.join(args.ckpt_dir, "ckpt.pt"), map_location="cpu")

    sae_mode = str(cfg_data.get("args", {}).get("sae_mode", "train"))
    if sae_mode == "identity":
        sae = IdentitySparseAutoencoder(embed_dim=embed_dim, topk=0).to(device).eval()
    else:
        latent_dim = int(cfg_data["args"].get("latent_dim", 256))
        topk = int(cfg_data["args"].get("topk", 16))
        sae = TopKSparseAutoencoder(SAEConfig(embed_dim=embed_dim, latent_dim=latent_dim, topk=topk))
        sae.load_state_dict(ckpt["sae"], strict=True)
        sae = sae.to(device).eval()

    masker = TEVIMasker(text_dim=embed_dim, latent_dim=sae.cfg.latent_dim)
    masker.load_state_dict(ckpt["masker"], strict=True)
    tevi = TEVI(sae=sae, masker=masker.to(device).eval())

    test_ds = ToyMultimodalDataset(toy_cfg, split="test", seed=int(cfg_data["args"]["seed"]))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    images = []
    texts = []
    for image, text in test_loader:
        images.append(image)
        texts.append(text)

    image = torch.cat(images, dim=0).to(device)
    text = torch.cat(texts, dim=0).to(device)

    base_sim = cosine_sim(image, text)
    tevi_sim = cosine_sim(tevi.edit(image, text), text)

    metrics = {
        "baseline": {
            "r@1": recall_at_k(base_sim, 1),
            "r@5": recall_at_k(base_sim, 5),
            "r@10": recall_at_k(base_sim, 10),
        },
        "tevi": {
            "r@1": recall_at_k(tevi_sim, 1),
            "r@5": recall_at_k(tevi_sim, 5),
            "r@10": recall_at_k(tevi_sim, 10),
        },
    }

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
