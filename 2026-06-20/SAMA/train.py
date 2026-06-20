"""Train SAMA: CME-MLLM adapters (Eqs. 7-9) + Anchor-Preserving Diffusion + ToyCLIP.

Saves cme.pt, diffusion.pt, clip.pt for test.py.
"""
from __future__ import annotations

import torch

from anchors import anchor_ids, entity_anchor_words
from data import STOI, TASKS, encode, split_train_test
from diffusion import PromptEncoder, TinyUNet, diffusion_loss
from filtering import ToyCLIP
from model import PAD, CME_MLLM, info_nce

torch.manual_seed(0)
GAMMA, BETA, OMEGA = 2.0, 0.1, 1.2          # Table II hyperparameters


def pad(seqs):
    L = max(len(s) for s in seqs)
    return torch.tensor([s + [PAD] * (L - len(s)) for s in seqs])


def batch(samples):
    ids = pad([encode(s.tokens) for s in samples])
    aids = pad([anchor_ids(s) for s in samples])
    imgs = torch.stack([s.image for s in samples])
    return ids, imgs, aids


def prompt_inputs(samples):
    """Anchor-weighted prompt (Eq. 10): entity-anchor tokens get weight OMEGA."""
    seqs = [encode(s.tokens) for s in samples]
    L = max(len(s) for s in seqs)
    ids, weights = [], []
    for s, seq in zip(samples, seqs):
        pinned = set(entity_anchor_words(s))
        w = [OMEGA if tok in pinned else 1.0 for tok in s.tokens]
        ids.append(seq + [PAD] * (L - len(seq)))
        weights.append(w + [0.0] * (L - len(w)))
    return torch.tensor(ids), torch.tensor(weights)


def task_accuracy(model, samples):
    """a_n in [0,1]: next-token accuracy on a task (drives w_n, Eq. 8)."""
    ids, imgs, aids = batch(samples)
    with torch.no_grad():
        logits, _, _, _ = model(ids, imgs, aids, samples[0].task)
    pred = logits[:, :-1].argmax(-1)
    tgt = ids[:, 1:]
    m = tgt != PAD
    return (pred[m] == tgt[m]).float().mean().item()


def main():
    train, _ = split_train_test(low_resource_frac=0.5)
    by_task = {t: [s for s in train if s.task == t] for t in TASKS}

    # --- 1. CME-MLLM (adapters) ----------------------------------------------
    cme = CME_MLLM()
    opt = torch.optim.Adam(cme.parameters(), lr=5e-3)
    for epoch in range(80):
        opt.zero_grad()
        total = 0.0
        us, ds = [], []
        for t in TASKS:
            samples = by_task[t]
            ids, imgs, aids = batch(samples)
            loss, u, d = cme.gen_loss(ids, imgs, aids, t)
            an = task_accuracy(cme, samples)
            wn = (1 - an) ** GAMMA                       # Eq. 8
            total = total + wn * loss
            us.append(u.mean(1)); ds.append(d.mean(1))
        u_all = torch.cat(us); d_all = torch.cat(ds)
        l_align = info_nce(u_all.unsqueeze(1), d_all.unsqueeze(1))  # Eq. 6
        total = total + BETA * l_align                  # Eq. 9
        total.backward()
        opt.step()
        if epoch % 20 == 0 or epoch == 79:
            accs = {t: round(task_accuracy(cme, by_task[t]), 2) for t in TASKS}
            print(f"[CME] epoch {epoch:3d}  L={total.item():.3f}  "
                  f"align={l_align.item():.3f}  acc={accs}")
    torch.save(cme.state_dict(), "cme.pt")

    # --- 2. Anchor-Preserving Diffusion --------------------------------------
    prompt_enc = PromptEncoder()
    unet = TinyUNet()
    opt_d = torch.optim.Adam(list(prompt_enc.parameters()) + list(unet.parameters()), lr=3e-3)
    p_ids, p_w = prompt_inputs(train)
    imgs = torch.stack([s.image for s in train])
    for step in range(300):
        opt_d.zero_grad()
        prompt = prompt_enc(p_ids, p_w)
        loss = diffusion_loss(unet, imgs, prompt)
        loss.backward()
        opt_d.step()
        if step % 100 == 0 or step == 299:
            print(f"[Diffusion] step {step:3d}  mse={loss.item():.4f}")
    torch.save({"prompt": prompt_enc.state_dict(), "unet": unet.state_dict()}, "diffusion.pt")

    # --- 3. ToyCLIP (for Dual-Constraint Filtering) --------------------------
    clip = ToyCLIP()
    opt_c = torch.optim.Adam(clip.parameters(), lr=3e-3)
    ids = pad([encode(s.tokens) for s in train])
    for step in range(200):
        opt_c.zero_grad()
        loss = clip.clip_loss(ids, imgs)
        loss.backward()
        opt_c.step()
        if step % 100 == 0 or step == 199:
            print(f"[ToyCLIP] step {step:3d}  loss={loss.item():.4f}")
    torch.save(clip.state_dict(), "clip.pt")
    print("saved -> cme.pt, diffusion.pt, clip.pt")


if __name__ == "__main__":
    main()
