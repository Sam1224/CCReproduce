from __future__ import annotations

import argparse
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kv_fold.kv_fold import KVFoldConfig, kv_fold_greedy_generate
from kv_fold.tasks.needle import build_needle_prompt


def _extract_number(text: str) -> str | None:
    m = re.search(r"\b\d{3,8}\b", text)
    return m.group(0) if m else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--target-tokens", type=int, default=1024)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()

    sample = build_needle_prompt(
        key="ECO_CONTENT_KEY",
        value="12345",
        target_tokens=args.target_tokens,
        tokenizer=tokenizer,
    )

    enc = tokenizer(sample.prompt, return_tensors="pt")
    input_ids = enc.input_ids
    attention_mask = enc.attention_mask

    out_ids = kv_fold_greedy_generate(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=args.max_new_tokens,
        cfg=KVFoldConfig(chunk_size=args.chunk_size),
    )

    gen_text = tokenizer.decode(out_ids[0][input_ids.shape[1] :], skip_special_tokens=True)
    extracted = _extract_number(gen_text)

    print("expected:", sample.value)
    print("generated:", gen_text.strip())
    print("extracted:", extracted)


if __name__ == "__main__":
    main()
