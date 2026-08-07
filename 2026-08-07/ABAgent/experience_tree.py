from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from dataset import HistoricalChunk, RequestCase


@dataclass(frozen=True)
class PathKey:
    domain: str
    scenario: str
    stage: str
    objective: str

    def as_tuple(self) -> Tuple[str, str, str, str]:
        return (self.domain, self.scenario, self.stage, self.objective)


class ExperienceTree:
    def __init__(self, chunks: Iterable[HistoricalChunk]) -> None:
        self.chunks = list(chunks)
        self.path_to_indices: Dict[Tuple[str, str, str, str], List[int]] = {}
        for index, chunk in enumerate(self.chunks):
            key = self.chunk_key(chunk).as_tuple()
            self.path_to_indices.setdefault(key, []).append(index)

    @staticmethod
    def chunk_key(chunk: HistoricalChunk) -> PathKey:
        return PathKey(chunk.domain, chunk.scenario, chunk.stage, chunk.objective)

    @staticmethod
    def request_key(request: RequestCase) -> PathKey:
        return PathKey(request.domain, request.scenario, request.stage, request.objective)

    def tree_distance(self, request: RequestCase, chunk: HistoricalChunk) -> int:
        request_key = self.request_key(request).as_tuple()
        chunk_key = self.chunk_key(chunk).as_tuple()
        return sum(1 for left, right in zip(request_key, chunk_key) if left != right)

    def tree_boost(self, request: RequestCase, chunk: HistoricalChunk) -> float:
        distance = self.tree_distance(request, chunk)
        stage_match = 1.0 if request.stage == chunk.stage else 0.0
        objective_match = 1.5 if request.objective == chunk.objective else 0.0
        scenario_match = 1.0 if request.scenario == chunk.scenario else 0.0
        return (3.5 - distance) + stage_match + objective_match + scenario_match

    def candidate_pool(self, request: RequestCase) -> List[HistoricalChunk]:
        exact = self.path_to_indices.get(self.request_key(request).as_tuple(), [])
        if exact:
            return [self.chunks[index] for index in exact]
        return self.chunks
