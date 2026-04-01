from __future__ import annotations

import argparse

import torch

from dataset import TEMPLATE_POOL, load_toy_dataset, tokenize
from model import ReLogClassifier
from relog import refine_log


def build_vocab_index(vocab: list[str]) -> dict[str, int]:
    return {token: idx for idx, token in enumerate(vocab)}


def encode(text: str, *, vocab_index: dict[str, int], max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = tokenize(text)
    ids = [vocab_index.get(t, vocab_index.get("<unk>", 1)) for t in tokens][:max_len]
    attn = [1] * len(ids)
    while len(ids) < max_len:
        ids.append(vocab_index.get("<pad>", 0))
        attn.append(0)
    return torch.tensor(ids, dtype=torch.long), torch.tensor(attn, dtype=torch.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt.pt")
    args = ap.parse_args()

    payload = torch.load(args.ckpt, map_location="cpu")
    vocab = payload["vocab"]
    max_len = int(payload.get("max_len") or 128)
    vocab_index = build_vocab_index(vocab)

    model = ReLogClassifier(vocab_size=len(vocab), num_levels=4, num_templates=len(TEMPLATE_POOL))
    model.load_state_dict(payload["state_dict"])

    examples = load_toy_dataset()
    correct_template = 0
    correct_level = 0

    for example in examples:
        text = example.code + "\n" + example.runtime_trace
        token_ids, attn = encode(text, vocab_index=vocab_index, max_len=max_len)
        pred = refine_log(model, example=example, token_ids=token_ids, attention_mask=attn)

        if pred.template == example.target_template:
            correct_template += 1
        if pred.level == int(example.target_level):
            correct_level += 1

        print("---")
        print(example.runtime_trace)
        print("pred:", pred)
        print("gold:", example.target_level, example.target_template)

    print("template_acc", correct_template / max(1, len(examples)))
    print("level_acc", correct_level / max(1, len(examples)))


if __name__ == "__main__":
    main()
