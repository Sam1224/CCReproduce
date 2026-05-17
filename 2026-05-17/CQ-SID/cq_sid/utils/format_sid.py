from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SID:
    category_id: int
    codes: tuple[int, ...]

    def to_tokens(self) -> list[str]:
        return [f"C{self.category_id}"] + [f"Q{c}" for c in self.codes]

    @classmethod
    def from_tokens(cls, tokens: list[str]) -> "SID":
        if not tokens or not tokens[0].startswith("C"):
            raise ValueError(f"invalid SID tokens: {tokens}")
        category_id = int(tokens[0][1:])
        codes = []
        for t in tokens[1:]:
            if not t.startswith("Q"):
                raise ValueError(f"invalid SID token: {t}")
            codes.append(int(t[1:]))
        return cls(category_id=category_id, codes=tuple(codes))
