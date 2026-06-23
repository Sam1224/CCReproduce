"""
train_piopd.py — Stage 3: Preference-Internalized On-Policy Distillation
(paper Sec. 3.4.3, Algorithm 1, Eq. 11-16).

Key idea: NO separately-trained reward model. A frozen *teacher* sees a
posterior-augmented prompt x^(T) = x ⊕_rand y_ref (the clicked query inserted
at a random field slot — Randomized Context Augmentation). The *student* sees
only the standard prompt x^(S) = x. We sample an on-policy rollout from the
student and match the teacher's soft token distributions on the student-visited
prefixes via a bidirectional KL (FKL + RKL), plus an SFT anchor on y_ref,
plus an entropy regulariser (anti top-1 collapse) and optional R-Drop / FGM.

    L_PIOPD = L_SFT + L_Distill + λ_E·L_Ent + λ_RD·L_RDrop + λ_FGM·L_FGM

After training the teacher and y_ref are discarded; only the student deploys.

Example (smoke test):
    python train_piopd.py --tiny --max_steps 3 --save_dir outputs/piopd
"""

from __future__ import annotations

import argparse
import copy
import os

import torch
from torch.utils.data import DataLoader

from data import generate_dataset, PIOPDDataset
from model import OneBarGenerator
from losses import piopd_loss, sft_loss


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="facebook/bart-base")
    ap.add_argument("--init_from", default="",
                    help="Stage-2 checkpoint dir (else fresh / tiny model)")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--n_train", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--lambda_fkl", type=float, default=1.0)
    ap.add_argument("--lambda_rkl", type=float, default=1.0)
    ap.add_argument("--lambda_ent", type=float, default=0.01)
    ap.add_argument("--lambda_rdrop", type=float, default=0.0)
    ap.add_argument("--lambda_fgm", type=float, default=0.0)
    ap.add_argument("--fgm_eps", type=float, default=1.0)
    ap.add_argument("--rollout_len", type=int, default=24)
    ap.add_argument("--prompt_style", default="compressed")
    ap.add_argument("--save_dir", default="outputs/piopd")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def build_model(args):
    name = args.init_from if args.init_from else args.model_name
    return OneBarGenerator(name, tiny=args.tiny, offline=args.offline)


def fgm_attack(model, eps, embed_attr="model.shared"):
    """FGM (Eq. 15): perturb input embeddings along the gradient direction.

    Δe = eps · ∇_e L_task / ||∇_e L_task||
    Returns a restore() closure. Call AFTER a backward pass that populated
    the embedding grad. (Simplified single-layer FGM on the shared embedding.)
    """
    emb = model.model.get_input_embeddings()
    backup = emb.weight.data.clone()
    if emb.weight.grad is not None:
        norm = emb.weight.grad.norm()
        if norm != 0 and not torch.isnan(norm):
            emb.weight.data.add_(eps * emb.weight.grad / norm)

    def restore():
        emb.weight.data = backup
    return restore


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    student = build_model(args)
    tok = student.tokenizer
    pad_id = tok.pad_token_id

    # frozen teacher = copy of the Stage-2 checkpoint (Algorithm 1, line 2)
    teacher = copy.deepcopy(student)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()

    samples = generate_dataset(n=args.n_train, seed=args.seed)
    ds = PIOPDDataset(samples, tok, prompt_style=args.prompt_style)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr)
    student.train()

    step = 0
    for epoch in range(args.epochs):
        for batch in dl:
            x_S = batch["student_input_ids"]
            m_S = batch["student_attention_mask"]
            x_T = batch["teacher_input_ids"]
            m_T = batch["teacher_attention_mask"]
            labels = batch["labels"]

            # ---- Algorithm 1, line 7: on-policy rollout from the student ----
            with torch.no_grad():
                y_roll = student.sample_rollout(x_S, m_S, max_len=args.rollout_len)
            rollout_mask = (y_roll != pad_id).float()

            # ---- line 8/9: student & teacher rollout logits (same decoder) ----
            student_rollout_logits = student.logits_for_sequence(x_S, m_S, y_roll)
            with torch.no_grad():
                teacher_rollout_logits = teacher.logits_for_sequence(x_T, m_T, y_roll)

            # ---- line 10: student label logits (CE anchor on y_ref) ----
            out_label = student(input_ids=x_S, attention_mask=m_S, labels=labels)
            student_label_logits = out_label.logits

            # optional 2nd dropout pass for R-Drop
            student_rollout_logits_2 = None
            if args.lambda_rdrop > 0:
                student_rollout_logits_2 = student.logits_for_sequence(x_S, m_S, y_roll)

            # ---- line 11: combined PIOPD loss (Eq. 16) ----
            total, comp = piopd_loss(
                student_label_logits, labels,
                student_rollout_logits, teacher_rollout_logits, rollout_mask,
                student_rollout_logits_2=student_rollout_logits_2,
                tau=args.tau, lambda_fkl=args.lambda_fkl,
                lambda_rkl=args.lambda_rkl, lambda_ent=args.lambda_ent,
                lambda_rdrop=args.lambda_rdrop)

            opt.zero_grad()
            total.backward()

            # ---- optional FGM input-space smoothing (Eq. 15) ----
            if args.lambda_fgm > 0:
                restore = fgm_attack(student, args.fgm_eps)
                adv_out = student(input_ids=x_S, attention_mask=m_S, labels=labels)
                adv_loss = args.lambda_fgm * sft_loss(adv_out.logits, labels)
                adv_loss.backward()
                restore()

            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            step += 1
            print(f"[PIOPD] ep {epoch} step {step} total {total.item():.4f} "
                  f"sft {comp['sft']:.3f} distill {comp['distill']:.3f} "
                  f"fkl {comp['fkl']:.3f} rkl {comp['rkl']:.3f} "
                  f"ent {comp['ent']:.3f}")
            if args.max_steps > 0 and step >= args.max_steps:
                break
        if args.max_steps > 0 and step >= args.max_steps:
            break

    # Algorithm 1, line 14: discard teacher & posterior signal; keep student.
    os.makedirs(args.save_dir, exist_ok=True)
    student.model.save_pretrained(args.save_dir)
    tok.save_pretrained(args.save_dir)
    print(f"[PIOPD] saved deployable student -> {args.save_dir}")


if __name__ == "__main__":
    main()
