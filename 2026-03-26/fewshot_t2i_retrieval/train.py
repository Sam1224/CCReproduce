from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from dataset import Pairs, make_pairs
from model import Encoder, clip_loss


def recall_at_1(img_emb: torch.Tensor, txt_emb: torch.Tensor) -> float:
    sims = img_emb @ txt_emb.t()
    pred = sims.argmax(dim=-1)
    gold = torch.arange(sims.shape[0])
    return (pred == gold).float().mean().item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrain-epochs", type=int, default=10)
    ap.add_argument("--adapt-epochs", type=int, default=5)
    ap.add_argument("--k", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # base pretrain
    base = make_pairs(n=2000, seed=1, num_classes=20)
    img_enc = Encoder().to(device)
    txt_enc = Encoder().to(device)
    opt = torch.optim.AdamW(list(img_enc.parameters()) + list(txt_enc.parameters()), lr=3e-3)

    for ep in range(args.pretrain_epochs):
        img_enc.train(); txt_enc.train()
        loss = clip_loss(img_enc(base.img.to(device)), txt_enc(base.txt.to(device)))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        print(f"pretrain epoch={ep} loss={loss.item():.4f}")

    # few-shot adaptation on novel classes
    novel = make_pairs(n=600, seed=2, num_classes=6)
    # take k shots per class (paired examples)
    keep = []
    for c in range(6):
        idx = (novel.cls == c).nonzero(as_tuple=False).view(-1)[: args.k]
        keep.append(idx)
    keep = torch.cat(keep)
    shots = Pairs(img=novel.img[keep], txt=novel.txt[keep], cls=novel.cls[keep])

    opt2 = torch.optim.AdamW(list(img_enc.parameters()) + list(txt_enc.parameters()), lr=1e-3)
    for ep in range(args.adapt_epochs):
        img_enc.train(); txt_enc.train()
        loss = clip_loss(img_enc(shots.img.to(device)), txt_enc(shots.txt.to(device)))
        opt2.zero_grad(set_to_none=True)
        loss.backward()
        opt2.step()
        print(f"adapt epoch={ep} loss={loss.item():.4f}")

    ckpt = Path("checkpoints")
    ckpt.mkdir(exist_ok=True)
    torch.save({"img": img_enc.state_dict(), "txt": txt_enc.state_dict()}, ckpt / "fewshot_t2i.pt")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
