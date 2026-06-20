"""Test SAMA: generate augmented samples, run Dual-Constraint Filtering, and report
a toy downstream macro-F1 (low-resource backbone vs backbone + SAMA augmentation).
"""
from __future__ import annotations

import torch

from anchors import anchor_ids
from data import (GOLD2ID, IMG_C, IMG_H, IMG_W, STOI, anchor_mask, encode,
                  split_train_test)
from diffusion import PromptEncoder, TinyUNet, sample as diff_sample
from filtering import ToyCLIP, dual_constraint_select
from generate import decode, generate_text
from model import PAD, CME_MLLM, MIEClassifier
from train import prompt_inputs

torch.manual_seed(1)
K = 5                                  # candidates per sample
PADID = PAD


def load():
    cme = CME_MLLM(); cme.load_state_dict(torch.load("cme.pt")); cme.eval()
    d = torch.load("diffusion.pt")
    penc = PromptEncoder(); penc.load_state_dict(d["prompt"]); penc.eval()
    unet = TinyUNet(); unet.load_state_dict(d["unet"]); unet.eval()
    clip = ToyCLIP(); clip.load_state_dict(torch.load("clip.pt")); clip.eval()
    return cme, penc, unet, clip


def augment(cme, penc, unet, clip, samples):
    """Generate K candidates/sample, filter, return kept synthetic samples."""
    mask = anchor_mask().unsqueeze(0)                     # [1,1,8,8]
    syn = []
    kept_print = []
    for s in samples:
        p_ids, p_w = prompt_inputs([s])
        prompt = penc(p_ids, p_w)
        cand_text, cand_imgs = [], []
        for _ in range(K):
            cand_text.append(generate_text(cme, s))
            img = diff_sample(unet, s.image.unsqueeze(0), prompt, mask, strength=0.8)
            cand_imgs.append(img[0])
        cand_imgs = torch.stack(cand_imgs)
        best, scores = dual_constraint_select(
            clip, cand_text, cand_imgs, anchor_ids(s), alpha=0.6, tau=0.75)
        if best is None:
            continue
        # label projection: synthetic pair inherits the original gold label
        syn.append((cand_text[best], cand_imgs[best], GOLD2ID[s.gold_type]))
        if len(kept_print) < 4:
            kept_print.append((s, decode(cand_text[best]), scores[best]))
    return syn, kept_print


def _pad(seqs):
    L = max(len(s) for s in seqs)
    return torch.tensor([s + [PADID] * (L - len(s)) for s in seqs])


def macro_f1(model, test):
    ids = _pad([encode(s.tokens) for s in test])
    imgs = torch.stack([s.image for s in test])
    gold = torch.tensor([GOLD2ID[s.gold_type] for s in test])
    with torch.no_grad():
        pred = model(ids, imgs).argmax(-1)
    f1s = []
    for c in range(len(GOLD2ID)):
        tp = int(((pred == c) & (gold == c)).sum())
        fp = int(((pred == c) & (gold != c)).sum())
        fn = int(((pred != c) & (gold == c)).sum())
        if tp + fp + fn == 0:
            continue
        p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9)
        f1s.append(2 * p * r / (p + r + 1e-9))
    return sum(f1s) / len(f1s)


def train_backbone(train_items, epochs=150, lr=5e-3):
    model = MIEClassifier()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ids = _pad([it[0] for it in train_items])
    imgs = torch.stack([it[1] for it in train_items])
    gold = torch.tensor([it[2] for it in train_items])
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(ids, imgs), gold)
        loss.backward(); opt.step()
    return model


def main():
    train, test = split_train_test(low_resource_frac=0.5)
    cme, penc, unet, clip = load()

    print("=== SAMA augmentation + Dual-Constraint Filtering ===")
    syn, kept = augment(cme, penc, unet, clip, train)
    print(f"generated candidates: {len(train)} samples x K={K}")
    print(f"kept after dual-constraint filtering (tau=0.75): {len(syn)} synthetic samples\n")
    print("--- sample synthetic texts (anchor-preserving) ---")
    for s, txt, sc in kept:
        print(f"  [{s.task}] orig: {' '.join(s.tokens)}")
        print(f"         syn : {txt}   (S_conf={sc:.3f})")

    # downstream backbone: original low-resource vs + SAMA synthetic ----------
    orig_items = [(encode(s.tokens), s.image, GOLD2ID[s.gold_type]) for s in train]
    base = train_backbone(orig_items)
    aug = train_backbone(orig_items + syn)
    f1_base = macro_f1(base, test)
    f1_aug = macro_f1(aug, test)
    print("\n--- toy downstream macro-F1 (test set) ---")
    print(f"  backbone (low-resource only)     : {f1_base:.3f}")
    print(f"  backbone + SAMA augmentation     : {f1_aug:.3f}")
    print(f"  delta                            : {f1_aug - f1_base:+.3f}")


if __name__ == "__main__":
    main()
