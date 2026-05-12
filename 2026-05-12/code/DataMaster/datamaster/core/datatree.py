from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable

from .memory import GlobalMemory
from .ops import AddSourceOp, op_dedup, op_drop_label_conflicts, op_filter_min_tokens
from .pool import DataPool
from .state import DataState


@dataclass
class Node:
    node_id: int
    node_type: str  # "red" (discover/compose) or "black" (clean/transform)
    depth: int
    state: DataState
    parent_id: int | None
    action: str
    score: float
    metrics: dict[str, float]


@dataclass
class SearchConfig:
    max_depth: int = 3
    beam_size: int = 6
    max_expansions: int = 30
    max_children_per_node: int = 6
    artifacts_dir: str = "artifacts"


class DataMasterSearch:
    def __init__(
        self,
        pool: DataPool,
        evaluate: Callable[[DataState], tuple[float, dict[str, float]]],
        config: SearchConfig | None = None,
    ) -> None:
        self.pool = pool
        self.evaluate = evaluate
        self.config = config or SearchConfig()
        self.memory = GlobalMemory()
        self._next_id = 1

    def _new_id(self) -> int:
        nid = self._next_id
        self._next_id += 1
        return nid

    def _log(self, node: Node) -> None:
        self.memory.log_node(
            {
                "node_id": node.node_id,
                "parent_id": node.parent_id,
                "node_type": node.node_type,
                "depth": node.depth,
                "action": node.action,
                "score": node.score,
                "metrics": node.metrics,
                "signature": node.state.signature(),
                "sources": list(node.state.sources),
                "ops": list(node.state.ops),
                "num_records": len(node.state.records),
            }
        )

    def _maybe_record(self, node: Node) -> bool:
        sig = node.state.signature()
        if self.memory.seen(sig):
            return False
        self.memory.mark(sig)
        self._log(node)
        return True

    def _expand_red(self, node: Node) -> list[tuple[str, DataState]]:
        candidates: list[tuple[str, DataState]] = []
        for source in self.pool.list_sources():
            if source in node.state.sources:
                continue
            if source == "base":
                continue
            op = AddSourceOp(source)
            next_state = op.apply(node.state, self.pool)
            candidates.append((op.name(), next_state))
        return candidates[: self.config.max_children_per_node]

    def _expand_black(self, node: Node) -> list[tuple[str, DataState]]:
        ops: list[tuple[str, Callable]] = [
            ("dedup", op_dedup),
            ("filter_min_tokens:3", op_filter_min_tokens(3)),
            ("drop_label_conflicts", op_drop_label_conflicts),
        ]

        candidates: list[tuple[str, DataState]] = []
        for name, fn in ops:
            if any(o.startswith(name) for o in node.state.ops):
                continue
            next_state = node.state.clone()
            next_state.apply_inplace(name, fn)
            candidates.append((name, next_state))
        return candidates[: self.config.max_children_per_node]

    def _child_type(self, node_type: str) -> str:
        return "black" if node_type == "red" else "red"

    def run(self, initial_state: DataState) -> Node:
        os.makedirs(self.config.artifacts_dir, exist_ok=True)

        base_score, base_metrics = self.evaluate(initial_state)
        best = Node(
            node_id=self._new_id(),
            node_type="red",
            depth=0,
            state=initial_state,
            parent_id=None,
            action="init",
            score=base_score,
            metrics=base_metrics,
        )
        self._maybe_record(best)

        frontier: list[Node] = [best]
        expansions = 0

        while frontier and expansions < self.config.max_expansions:
            frontier.sort(key=lambda n: n.score, reverse=True)
            frontier = frontier[: self.config.beam_size]

            node = frontier.pop(0)
            if node.depth >= self.config.max_depth:
                continue

            expansions += 1
            child_type = self._child_type(node.node_type)

            if node.node_type == "red":
                children = self._expand_red(node)
            else:
                children = self._expand_black(node)

            for action, state in children:
                score, metrics = self.evaluate(state)
                child = Node(
                    node_id=self._new_id(),
                    node_type=child_type,
                    depth=node.depth + 1,
                    state=state,
                    parent_id=node.node_id,
                    action=action,
                    score=score,
                    metrics=metrics,
                )
                if not self._maybe_record(child):
                    continue
                frontier.append(child)
                if child.score > best.score:
                    best = child

        with open(os.path.join(self.config.artifacts_dir, "search_trace.json"), "w", encoding="utf-8") as f:
            json.dump({"best": best.node_id, "nodes": self.memory.node_logs}, f, ensure_ascii=False, indent=2)

        return best
