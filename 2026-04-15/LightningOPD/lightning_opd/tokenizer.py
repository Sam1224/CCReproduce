from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tokenizer:
    tokens: list[str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "token_to_id", {t: i for i, t in enumerate(self.tokens)})

    def encode(self, seq: list[str]) -> list[int]:
        return [self.token_to_id[t] for t in seq]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.tokens[i] for i in ids]

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)


def build_default_tokenizer() -> Tokenizer:
    special = ["<pad>", "<bos>", "<eos>"]
    ops = ["ADD", "+", "="]
    digits = [str(i) for i in range(10)]
    return Tokenizer(tokens=special + ops + digits)
