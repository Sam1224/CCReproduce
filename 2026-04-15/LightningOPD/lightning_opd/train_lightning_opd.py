from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import OPDDataset, OPDExample, collate_opd
from .model import GRULM
from .opd_math import logp_of_labels, shift_logits_and_labels
from .tokenizer import Tokenizer
from .utils import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sft_dir", type=str, required=True)
    p.add_argument("--opd_dataset", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--tau", type=float, default=2.0)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def load_model(ckpt_path: Path, device: str) -> tuple[GRULM, dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    meta = ckpt["meta"]
    cfg = meta["model"]
    model = GRULM(type("Cfg", (), cfg))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    return model, meta


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sft_dir = Path(args.sft_dir)
    model, meta = load_model(sft_dir / "model.pt", args.device)
    tok = Tokenizer(tokens=meta["tokenizer"])

    raw = torch.load(args.opd_dataset, map_location="cpu")
    examples = [
        OPDExample(prompt=p, response=r, teacher_logp=lp)
        for p, r, lp in zip(raw["prompts"], raw["responses"], raw["teacher_logp"])
    ]
    ds = OPDDataset(examples)

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_opd(b, tok),
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_tokens = 0

        for batch, teacher_logp in loader:
            input_ids = batch.input_ids.to(args.device)
            loss_mask = batch.loss_mask.to(args.device)
            teacher_logp = teacher_logp.to(args.device)

            logits = model(input_ids)
            shift_logits, shift_labels = shift_logits_and_labels(logits, input_ids)
            shift_mask = loss_mask[:, 1:]
            student_logp = logp_of_labels(shift_logits, shift_labels)  # [B, T-1]

            # Align teacher logp to shifted positions
            teacher_shift = teacher_logp[:, 1:]

            # Practical OPD-style update: advantage is stop-grad, weight the logp gradient.
            advantage = (teacher_shift - student_logp.detach()).clamp(-args.tau, args.tau)
            loss = -(advantage[shift_mask] * student_logp[shift_mask]).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += float(loss.item()) * int(shift_mask.sum().item())
            total_tokens += int(shift_mask.sum().item())

        print(f"[LightningOPD] epoch={epoch+1} loss={total_loss/max(total_tokens,1):.6f}")

    save_meta = {
        **meta,
        "opd_dataset": str(args.opd_dataset),
        "tau": args.tau,
        "algo": "toy-lightning-opd",
    }
    (out_dir / "meta.json").write_text(json.dumps(save_meta, indent=2), encoding="utf-8")
    torch.save({"state_dict": model.state_dict(), "meta": save_meta}, out_dir / "model.pt")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
