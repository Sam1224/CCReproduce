from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


Record = dict[str, Any]  # expects keys: text(str), label(int), source(str)


@dataclass
class DataState:
    records: list[Record]
    sources: list[str] = field(default_factory=list)
    ops: list[str] = field(default_factory=list)

    def clone(self) -> "DataState":
        return DataState(records=[dict(r) for r in self.records], sources=list(self.sources), ops=list(self.ops))

    def apply_inplace(self, op_name: str, fn: Callable[[list[Record]], list[Record]]) -> None:
        self.records = fn(self.records)
        self.ops.append(op_name)

    def signature(self) -> str:
        sources_part = "+".join(sorted(self.sources))
        ops_part = "+".join(self.ops)
        return f"sources={sources_part}::ops={ops_part}"
