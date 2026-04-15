from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .tokenizer import Tokenizer


@dataclass(frozen=True)
class Teacher:
    tokenizer: Tokenizer
    teacher_id: str  # "A" or "B"
    correct_p: float = 0.90

    def _target_response_tokens(self, prompt_tokens: list[str]) -> list[str]:
        # prompt: ["ADD", a, "+", b, "="]
        a = int(prompt_tokens[1])
        b = int(prompt_tokens[3])
        res = a + b

        # Teacher mismatch simulation:
        # teacher B learns a biased arithmetic (a+b+1) mod 19.
        if self.teacher_id.upper() == "B":
            res = (res + 1) % 19

        digits = list(str(res))
        return digits + ["<eos>"]

    def logp_next(self, prompt_tokens: list[str], prefix_tokens: list[str]) -> torch.Tensor:
        """Return log-prob over vocab for next token.

        The teacher is a simple rule-based distribution: assigns high probability to the
        next correct token in the target response; assigns uniform mass to others.
        """

        target = self._target_response_tokens(prompt_tokens)
        step = len(prefix_tokens)
        next_tok = target[step] if step < len(target) else "<eos>"

        vocab = self.tokenizer.vocab_size
        probs = torch.full((vocab,), (1.0 - self.correct_p) / (vocab - 1), dtype=torch.float32)
        probs[self.tokenizer.token_to_id[next_tok]] = self.correct_p
        return torch.log(probs)

    def logp_sequence(self, prompt_tokens: list[str], response_tokens: list[str]) -> torch.Tensor:
        """Return per-token logp of the response sequence under the teacher."""

        out = []
        prefix: list[str] = []
        for tok in response_tokens:
            lp = self.logp_next(prompt_tokens, prefix)
            out.append(lp[self.tokenizer.token_to_id[tok]])
            prefix.append(tok)
        return torch.stack(out, dim=0)  # [T_resp]
