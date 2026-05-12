from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GlobalMemory:
    node_logs: list[dict[str, Any]] = field(default_factory=list)
    visited_signatures: set[str] = field(default_factory=set)

    def seen(self, signature: str) -> bool:
        return signature in self.visited_signatures

    def mark(self, signature: str) -> None:
        self.visited_signatures.add(signature)

    def log_node(self, payload: dict[str, Any]) -> None:
        self.node_logs.append(payload)
