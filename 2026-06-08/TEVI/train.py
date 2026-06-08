from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ToyConfig, ToyMultimodalDataset
from metrics import cosine_sim, recall_at_k
from models import (
    IdentitySparseAutoencoder,
    SAEConfig,
    TEVI,
    TEVIMasker,
    TopKSparseAutoencoder,
    info_nce_loss,
)


def build_sae(
    device: torch.device,
    train_loader: DataLoader,
    embed_dim: int,
    latent_dim: int,
    topk: int,
    lr: float,
    epochs: int,
    mode: str,
) -> torch.nn.Module:
    if mode == "identity":
        return IdentitySparseAutoencoder(embed_dim=embed_dim, topk=0).to(device)

    sae = TopKSparseAutoencoder(SAEConfig(embed_dim=embed_dim, latent_dim=latent_dim, topk=topk)).to(device)
    opt = torch.optim.AdamW(sae.parameters(), lr=lr, weight_decay=1e-4)

    for _ in range(epochs):
        sae.train()
        for image, _text in train_loader:
            image = image.to(device)
            x_hat, z = sae(image)
            recon = F.mse_loss(x_hat, image)
            l1 = z.abs().mean()
            loss = recon + 1e-3 * l1
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    return sae


def train_tevi(
    device: torch.device,
    train_loader: DataLoader,
    sae: torch.nn.Module,
    lr: float,
    epochs: int,
) -> TEVI:
    sae.eval()
    for p in sae.parameters():
        p.requires_grad = False

    masker = TEVIMasker(text_dim=sae.cfg.embed_dim, latent_dim=sae.cfg.latent_dim).to(device)
    tevi = TEVI(sae=sae, masker=masker).to(device)
    opt = torch.optim.AdamW(masker.parameters(), lr=lr, weight_decay=1e-4)

    for _ in range(epochs):
        tevi.train()
        for image, text in train_loader:
            image = image.to(device)
            text = text.to(device)

            edited = tevi.edit(image, text)
            sim = cosine_sim(edited, text)
            loss = info_nce_loss(sim)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    return tevi


@torch.no_grad()
def quick_eval(device: torch.device, tevi: TEVI, test_loader: DataLoader) -> dict:
    tevi.eval()

    images = []
    texts = []
    for image, text in test_loader:
        images.append(image)
        texts.append(text)

    image = torch.cat(images, dim=0).to(device)
    text = torch.cat(texts, dim=0).to(device)

    base_sim = cosine_sim(image, text)
    tevi_sim = cosine_sim(tevi.edit(image, text), text)

    return {
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num_samples", type=int, default=4000)
    parser.add_argument("--num_concepts", type=int, default=64)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--concepts_per_caption", type=int, default=6)
    parser.add_argument("--nuisance_per_image", type=int, default=20)

    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--sae_mode", choices=["identity", "train"], default="identity")

    parser.add_argument("--sae_lr", type=float, default=3e-4)
    parser.add_argument("--sae_epochs", type=int, default=10)

    parser.add_argument("--tevi_lr", type=float, default=5e-4)
    parser.add_argument("--tevi_epochs", type=int, default=10)

    parser.add_argument("--batch_size", type=int, default=256)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = ToyConfig(
        num_samples=args.num_samples,
        num_concepts=args.num_concepts,
        embed_dim=args.embed_dim,
        concepts_per_caption=args.concepts_per_caption,
        nuisance_per_image=args.nuisance_per_image,
    )

    train_ds = ToyMultimodalDataset(cfg, split="train", seed=args.seed)
    test_ds = ToyMultimodalDataset(cfg, split="test", seed=args.seed)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    sae = build_sae(
        device=device,
        train_loader=train_loader,
        embed_dim=cfg.embed_dim,
        latent_dim=args.latent_dim,
        topk=args.topk,
        lr=args.sae_lr,
        epochs=args.sae_epochs,
        mode=args.sae_mode,
    )

    tevi = train_tevi(
        device=device,
        train_loader=train_loader,
        sae=sae,
        lr=args.tevi_lr,
        epochs=args.tevi_epochs,
    )

    metrics = quick_eval(device, tevi, test_loader)

    torch.save({"sae": sae.state_dict(), "masker": tevi.masker.state_dict()}, os.path.join(args.out_dir, "ckpt.pt"))

    with open(os.path.join(args.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"toy": asdict(cfg), "args": vars(args)}, f, ensure_ascii=False, indent=2)

    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
