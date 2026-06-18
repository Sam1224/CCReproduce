from __future__ import annotations

"""Training pipeline for ProductConsistency reproduction.

Two stages, mirroring the paper:

  Stage 0 (captioner warmup): train the re-caption/OCR module so it can read
           (brand, text) — analogous to the paper relying on a strong captioner+OCR.
  Stage 1 (SFT): supervised fine-tuning of the editor to reconstruct the GT edited
           target (identity-preserving edit). Saves editor_sft.pt.
  Stage 2 (RL): policy-gradient (REINFORCE w/ baseline) on the *sampled* editor using
           the Cyclic Consistency reward (+ small visual-consistency term), exactly the
           contribution of the paper. Saves editor_rl.pt.

Run: python train.py            # quick default (CPU friendly)
     python train.py --epochs 4 --rl-steps 300
"""

import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import ToyProductDataset, collate, _LO, _HI
from model import PCConfig, ProductEditor, ProductCaptioner, cyclic_consistency_reward


def product_region(x):
    return x[:, :, _LO:_HI, _LO:_HI]


def train_captioner(captioner, loader, device, epochs=3, lr=2e-3):
    opt = torch.optim.Adam(captioner.parameters(), lr=lr)
    captioner.train()
    for ep in range(epochs):
        tot = 0.0
        for b in loader:
            src = b["src"].to(device)
            brand_logits, text_logits = captioner(src)
            loss = F.cross_entropy(brand_logits, b["brand"].to(device))
            loss = loss + F.cross_entropy(
                text_logits.reshape(-1, text_logits.size(-1)),
                b["text"].to(device).reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        print(f"[captioner] epoch {ep+1}/{epochs} loss {tot/len(loader):.4f}")


def train_sft(editor, loader, device, epochs=3, lr=2e-3):
    opt = torch.optim.Adam(editor.parameters(), lr=lr)
    editor.train()
    for ep in range(epochs):
        tot = 0.0
        for b in loader:
            src, tgt, instr = b["src"].to(device), b["tgt"].to(device), b["instr"].to(device)
            edited, _ = editor(src, instr, sample=False)
            loss = F.l1_loss(edited, tgt)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        print(f"[SFT] epoch {ep+1}/{epochs} L1 {tot/len(loader):.4f}")


def train_rl(editor, captioner, loader, device, steps=200, lr=5e-4):
    """REINFORCE with a moving-average baseline on the cyclic-consistency reward."""
    opt = torch.optim.Adam(editor.parameters(), lr=lr)
    for p in captioner.parameters():
        p.requires_grad_(False)
    captioner.eval()
    editor.train()
    baseline = 0.0
    it = iter(loader)
    for step in range(steps):
        try:
            b = next(it)
        except StopIteration:
            it = iter(loader); b = next(it)
        src, instr = b["src"].to(device), b["instr"].to(device)
        tgt = b["tgt"].to(device)
        desc = {"brand": b["brand"].to(device), "text": b["text"].to(device)}

        edited, logp = editor(src, instr, sample=True)
        with torch.no_grad():
            r_cyc = cyclic_consistency_reward(captioner, edited, desc)        # identity
            # visual-consistency term: product region close to source product region
            r_vis = 1.0 - (product_region(edited) - product_region(src)).abs().mean((1, 2, 3))
            reward = 0.7 * r_cyc + 0.3 * r_vis
        baseline = 0.9 * baseline + 0.1 * reward.mean().item()
        adv = reward - baseline
        pg_loss = -(adv.detach() * logp).mean()
        # light reconstruction anchor keeps backgrounds realistic (prevents reward hacking)
        anchor = 0.1 * F.l1_loss(edited, tgt)
        loss = pg_loss + anchor
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 50 == 0:
            print(f"[RL] step {step+1}/{steps} reward {reward.mean().item():.4f} "
                  f"(cyc {r_cyc.mean().item():.4f}) baseline {baseline:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--rl-steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--n", type=int, default=512)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    cfg = PCConfig()

    ds = ToyProductDataset(n=args.n, seed=0)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=collate)

    captioner = ProductCaptioner(cfg).to(device)
    editor = ProductEditor(cfg).to(device)

    print("== Stage 0: captioner warmup ==")
    train_captioner(captioner, loader, device, epochs=args.epochs)
    torch.save(captioner.state_dict(), "captioner.pt")

    print("== Stage 1: SFT ==")
    train_sft(editor, loader, device, epochs=args.epochs)
    torch.save(editor.state_dict(), "editor_sft.pt")

    print("== Stage 2: RL with Cyclic Consistency reward ==")
    train_rl(editor, captioner, loader, device, steps=args.rl_steps)
    torch.save(editor.state_dict(), "editor_rl.pt")
    print("saved: captioner.pt, editor_sft.pt, editor_rl.pt")


if __name__ == "__main__":
    main()
