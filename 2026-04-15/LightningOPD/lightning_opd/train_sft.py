from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import collate_sft, make_sft_dataset
from .model import GRULM, ModelConfig, greedy_generate
from .opd_math import logp_of_labels, shift_logits_and_labels
from .teacher import Teacher
from .tokenizer import build_default_tokenizer
from .utils import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--teacher_id", type=str, default="A")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_size", type=int, default=2000)
    p.add_argument("--val_size", type=int, default=400)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


@torch.no_grad()
def eval_accuracy(model: GRULM, teacher: Teacher, n: int, device: str) -> float:
    tokenizer = teacher.tokenizer
    eos_id = tokenizer.token_to_id["<eos>"]

    correct = 0
    for a in range(10):
        for b in range(10):
            prompt = ["ADD", str(a), "+", str(b), "="]
            prompt_ids = torch.tensor([tokenizer.encode(prompt + ["<bos>"])], dtype=torch.long, device=device)
            out = greedy_generate(model, prompt_ids, eos_id=eos_id, max_new_tokens=3)[0].tolist()

            # decode response part
            decoded = tokenizer.decode(out)
            resp = []
            for t in decoded[len(prompt) + 1 :]:
                if t == "<eos>":
                    break
                resp.append(t)

            target = teacher._target_response_tokens(prompt)
            target_resp = [t for t in target if t != "<eos>"]
            if resp == target_resp:
                correct += 1

    return correct / 100.0


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    tokenizer = build_default_tokenizer()
    teacher = Teacher(tokenizer=tokenizer, teacher_id=args.teacher_id)

    train_ds = make_sft_dataset(tokenizer, teacher, n=args.train_size, seed=args.seed)
    val_ds = make_sft_dataset(tokenizer, teacher, n=args.val_size, seed=args.seed + 1)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_sft(b, tokenizer),
    )

    cfg = ModelConfig(vocab_size=tokenizer.vocab_size)
    model = GRULM(cfg).to(args.device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_tokens = 0

        for batch in train_loader:
            input_ids = batch.input_ids.to(args.device)
            loss_mask = batch.loss_mask.to(args.device)

            logits = model(input_ids)
            shift_logits, shift_labels = shift_logits_and_labels(logits, input_ids)
            shift_mask = loss_mask[:, 1:]

            token_logp = logp_of_labels(shift_logits, shift_labels)
            loss = -(token_logp[shift_mask]).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += float(loss.item()) * int(shift_mask.sum().item())
            total_tokens += int(shift_mask.sum().item())

        acc = eval_accuracy(model, teacher=teacher, n=args.val_size, device=args.device)
        ppl = float(torch.exp(torch.tensor(total_loss / max(total_tokens, 1))).item())
        print(f"[SFT] epoch={epoch+1} loss={total_loss/max(total_tokens,1):.4f} ppl={ppl:.2f} acc={acc:.3f}")

    meta = {
        "paper": "https://arxiv.org/abs/2604.13010",
        "teacher_id": args.teacher_id,
        "model": cfg.__dict__,
        "tokenizer": tokenizer.tokens,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    torch.save({"state_dict": model.state_dict(), "meta": meta}, out_dir / "model.pt")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
