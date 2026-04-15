from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .model import GRULM, greedy_generate
from .teacher import Teacher
from .tokenizer import Tokenizer
from .utils import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sft_dir", type=str, required=True)
    p.add_argument("--out_file", type=str, required=True)
    p.add_argument("--teacher_id", type=str, default="A")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--n_prompts", type=int, default=1000)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def load_sft_model(sft_dir: Path, device: str) -> tuple[GRULM, Tokenizer]:
    ckpt = torch.load(sft_dir / "model.pt", map_location=device)
    tokens = ckpt["meta"]["tokenizer"]
    tok = Tokenizer(tokens=tokens)

    cfg = ckpt["meta"]["model"]
    model = GRULM(type("Cfg", (), cfg))  # quick struct
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, tok


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    sft_dir = Path(args.sft_dir)
    model, tok = load_sft_model(sft_dir, args.device)

    teacher = Teacher(tokenizer=tok, teacher_id=args.teacher_id)
    eos_id = tok.token_to_id["<eos>"]

    prompts: list[list[str]] = []
    responses: list[list[str]] = []
    teacher_logps: list[torch.Tensor] = []

    g = torch.Generator().manual_seed(args.seed)

    for _ in range(args.n_prompts):
        a = int(torch.randint(0, 10, (1,), generator=g).item())
        b = int(torch.randint(0, 10, (1,), generator=g).item())
        prompt = ["ADD", str(a), "+", str(b), "="]

        prompt_ids = torch.tensor([tok.encode(prompt + ["<bos>"])], dtype=torch.long, device=args.device)
        out_ids = greedy_generate(model, prompt_ids, eos_id=eos_id, max_new_tokens=3)[0].tolist()
        out_tokens = tok.decode(out_ids)

        # response tokens after <bos>
        resp_tokens: list[str] = []
        for t in out_tokens[len(prompt) + 1 :]:
            resp_tokens.append(t)
            if t == "<eos>":
                break
        if resp_tokens[-1] != "<eos>":
            resp_tokens.append("<eos>")

        prompts.append(prompt)
        responses.append(resp_tokens)
        teacher_logps.append(teacher.logp_sequence(prompt, resp_tokens))

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "paper": "https://arxiv.org/abs/2604.13010",
            "teacher_id": args.teacher_id,
            "tokenizer": tok.tokens,
            "prompts": prompts,
            "responses": responses,
            "teacher_logp": teacher_logps,
        },
        out_path,
    )
    print(f"Saved OPD dataset to {out_path}")


if __name__ == "__main__":
    main()
