from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class RepoChunk:
    chunk_id: str
    path: str
    start_line: int
    end_line: int
    text: str


@dataclass
class QAExample:
    qid: str
    question: str
    answer: str
    support_chunk_id: str


@dataclass
class RepoQADataset:
    chunks: List[RepoChunk]
    examples: List[QAExample]

    def to_json(self) -> Dict[str, Any]:
        return {
            "chunks": [c.__dict__ for c in self.chunks],
            "examples": [e.__dict__ for e in self.examples],
        }

    @staticmethod
    def from_json(obj: Dict[str, Any]) -> "RepoQADataset":
        chunks = [RepoChunk(**c) for c in obj.get("chunks", [])]
        examples = [QAExample(**e) for e in obj.get("examples", [])]
        return RepoQADataset(chunks=chunks, examples=examples)


def load_dataset(path: str) -> RepoQADataset:
    with open(path, "r", encoding="utf-8") as f:
        return RepoQADataset.from_json(json.load(f))


def save_dataset(ds: RepoQADataset, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ds.to_json(), f, ensure_ascii=False, indent=2)
