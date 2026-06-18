from __future__ import annotations

"""Evaluation for the ProductConsistency reproduction.

Builds a held-out toy ProductConsistency *benchmark* and reports the same kinds of
metrics the paper highlights:

  * Brand-identity accuracy  : captioner reads the correct brand from the edited image
  * Text fidelity / CER      : per-cell character error rate (proxy for OCR CER)
  * Background-change rate    : how much the background actually changed (edit happened)

We compare the SFT-only editor vs the SFT+RL editor to show that the Cyclic
Consistency RL reward improves product identity / text fidelity (mirrors the paper's
~5x OCR CER reduction), while still performing the requested edit.

Run: python test.py            # expects editor_sft.pt, editor_rl.pt, captioner.pt
"""

import argparse

import torch
from torch.utils.data import DataLoader

from data import ToyProductDataset, collate, _LO, _HI
from model import PCConfig, ProductEditor, ProductCaptioner, TEXT_LEN


def region(x):
    return x[:, :, _LO:_HI, _LO:_HI]


@torch.no_grad()
def evaluate(editor, captioner, loader, device):
    editor.eval(); captioner.eval()
    n = 0
    brand_correct = 0
    char_err = 0
    char_tot = 0
    bg_change = 0.0
    for b in loader:
        src, instr = b["src"].to(device), b["instr"].to(device)
        brand, text = b["brand"].to(device), b["text"].to(device)
        edited, _ = editor(src, instr, sample=False)
        pb, pt = captioner(edited)
        brand_correct += (pb.argmax(1) == brand).sum().item()
        pred_text = pt.argmax(-1)                              # [B, TEXT_LEN]
        char_err += (pred_text != text).sum().item()
        char_tot += text.numel()
        # background = outside product region
        mask = torch.ones_like(src); mask[:, :, _LO:_HI, _LO:_HI] = 0
        bg_change += ((edited - src).abs() * mask).sum().item() / mask.sum().item() * src.size(0)
        n += src.size(0)
    return {
        "brand_acc": brand_correct / n,
        "text_cer": char_err / char_tot,
        "bg_change": bg_change / n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = PCConfig()

    ds = ToyProductDataset(n=args.n, seed=123)                 # unseen split
    loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate)

    captioner = ProductCaptioner(cfg).to(device)
    captioner.load_state_dict(torch.load("captioner.pt", map_location=device))

    editor_sft = ProductEditor(cfg).to(device)
    editor_sft.load_state_dict(torch.load("editor_sft.pt", map_location=device))
    editor_rl = ProductEditor(cfg).to(device)
    editor_rl.load_state_dict(torch.load("editor_rl.pt", map_location=device))

    m_sft = evaluate(editor_sft, captioner, loader, device)
    m_rl = evaluate(editor_rl, captioner, loader, device)

    def row(name, m):
        print(f"{name:>10} | brand_acc {m['brand_acc']*100:5.1f}% | "
              f"text_CER {m['text_cer']*100:5.2f}% | bg_change {m['bg_change']:.4f}")

    print("ProductConsistency Benchmark (toy, unseen split)")
    print("-" * 64)
    row("SFT", m_sft)
    row("SFT+RL", m_rl)
    cer_ratio = (m_sft["text_cer"] + 1e-9) / (m_rl["text_cer"] + 1e-9)
    print("-" * 64)
    print(f"OCR CER reduction (SFT -> SFT+RL): {cer_ratio:.2f}x  "
          f"(paper reports ~5x on Qwen-Image-Edit-2511)")


if __name__ == "__main__":
    main()
