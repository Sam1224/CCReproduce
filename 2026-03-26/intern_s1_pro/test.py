from __future__ import annotations

import os
from pathlib import Path

import torch

from dataset import CONCEPTS, make_pairs, prompt_tokens
from model import ImgEncoder, TxtEncoder


def main() -> None:
    ckpt = torch.load("checkpoints/intern_s1_pro.pt", map_location="cpu")
    img = ImgEncoder(); txt = TxtEncoder()
    img.load_state_dict(ckpt["img"], strict=True)
    txt.load_state_dict(ckpt["txt"], strict=True)
    img.eval(); txt.eval()

    test = make_pairs(n=1000, seed=2)
    prompts = prompt_tokens(seed=7)

    with torch.no_grad():
        img_emb = img(test.img)
        txt_emb = txt(prompts)
        sims = img_emb @ txt_emb.t()
        pred = sims.argmax(dim=-1)

    acc = (pred == test.y).float().mean().item()
    print(f"zero_shot_acc={acc:.3f} concepts={CONCEPTS}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
