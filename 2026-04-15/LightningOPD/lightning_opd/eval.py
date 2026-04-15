from __future__ import annotations

import argparse

import torch

from .model import GRULM, greedy_generate
from .teacher import Teacher
from .tokenizer import Tokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--teacher_id", type=str, default="A")
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.ckpt, map_location=args.device)
    meta = ckpt["meta"]

    tok = Tokenizer(tokens=meta["tokenizer"])
    model = GRULM(type("Cfg", (), meta["model"]))
    model.load_state_dict(ckpt["state_dict"])
    model.to(args.device)
    model.eval()

    teacher = Teacher(tokenizer=tok, teacher_id=args.teacher_id)
    eos_id = tok.token_to_id["<eos>"]

    correct = 0
    for a in range(10):
        for b in range(10):
            prompt = ["ADD", str(a), "+", str(b), "="]
            prompt_ids = torch.tensor([tok.encode(prompt + ["<bos>"])], dtype=torch.long, device=args.device)
            out_ids = greedy_generate(model, prompt_ids, eos_id=eos_id, max_new_tokens=3)[0].tolist()
            out_tokens = tok.decode(out_ids)

            resp_tokens: list[str] = []
            for t in out_tokens[len(prompt) + 1 :]:
                if t == "<eos>":
                    break
                resp_tokens.append(t)

            target = teacher._target_response_tokens(prompt)
            target_resp = [t for t in target if t != "<eos>"]

            if resp_tokens == target_resp:
                correct += 1

    acc = correct / 100.0
    print(f"Accuracy (exact match) = {acc:.3f}")


if __name__ == "__main__":
    main()
