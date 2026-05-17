from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Vocab:
    token_to_id: dict[str, int]
    id_to_token: list[str]

    @classmethod
    def from_tokens(cls, tokens: list[str]) -> "Vocab":
        uniq = []
        seen = set()
        for token in tokens:
            if token in seen:
                continue
            seen.add(token)
            uniq.append(token)
        return cls({token: i for i, token in enumerate(uniq)}, uniq)

    @classmethod
    def from_state(cls, state: dict) -> "Vocab":
        return cls(token_to_id=dict(state["token_to_id"]), id_to_token=list(state["id_to_token"]))

    def to_state(self) -> dict:
        return {"token_to_id": self.token_to_id, "id_to_token": self.id_to_token}

    def encode(self, tokens: list[str]) -> list[int]:
        unk = self.token_to_id["<unk>"]
        return [self.token_to_id.get(token, unk) for token in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.id_to_token[i] for i in ids]


def basic_tokenize(text: str) -> list[str]:
    return [token for token in text.lower().replace("/", " ").replace("-", " ").split() if token]
