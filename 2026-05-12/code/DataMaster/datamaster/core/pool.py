from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .state import Record


@dataclass
class DataPool:
    sources: dict[str, list[Record]]

    def list_sources(self) -> list[str]:
        return sorted(self.sources.keys())

    def get(self, name: str) -> list[Record]:
        if name not in self.sources:
            raise KeyError(f"unknown data source: {name}")
        return [dict(r) for r in self.sources[name]]
