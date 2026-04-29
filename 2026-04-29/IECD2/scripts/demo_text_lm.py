from __future__ import annotations

import argparse

import torch

from iecd2.core import IECDConfig, iecd_fuse_logits
from iecd2.hf_adapter import HFTextLM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="distilgpt2")
    parser.add_argument("--question", default="What is in the picture?")
    parser.add_argument("--eta", type=float, default=-3.0)
    args = parser.parse_args()

    lm = HFTextLM(model_name=args.model)

    instruction_prompt = (
        "You are a helpful assistant. Answer the question with a detailed sentence.\n"
        f"Question: {args.question}\nAnswer:"
    )
    evidence_prompt = (
        "You are a careful assistant. Only answer using directly observable evidence. "
        "If uncertain, answer 'I am not sure'.\n"
        f"Question: {args.question}\nAnswer:"
    )

    logits_i = lm.next_token_logits(instruction_prompt)
    logits_e = lm.next_token_logits(evidence_prompt)

    result = iecd_fuse_logits(logits_i, logits_e, config=IECDConfig(eta=args.eta))

    next_token_id = int(torch.argmax(result.fused_probs, dim=-1).item())
    next_token = lm.tokenizer.decode([next_token_id])

    print("gate_g:", float(result.gate_g.item()))
    print("symmetric_kl:", float(result.symmetric_kl.item()))
    print("next_token (greedy):", repr(next_token))


if __name__ == "__main__":
    main()
