from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kv_fold.kv_fold import KVFoldConfig, kv_fold_forward_logits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=64)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()

    text = (
        "This is a long document about e-commerce content understanding and governance. "
        * 200
    )
    input_ids = tokenizer(text, return_tensors="pt").input_ids[:, : args.seq_len]
    attention_mask = torch.ones_like(input_ids)

    with torch.inference_mode():
        full_logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
        fold_logits = kv_fold_forward_logits(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            cfg=KVFoldConfig(chunk_size=args.chunk_size),
        )

    max_abs = (full_logits - fold_logits).abs().max().item()
    print(f"max_abs_diff={max_abs:.6g}")

    torch.testing.assert_close(full_logits, fold_logits, atol=1e-4, rtol=1e-4)
    print("OK: KV-Fold logits match full forward (within tolerance)")


if __name__ == "__main__":
    main()
