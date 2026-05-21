from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Task:
    """A minimal interactive task instance.

    The downstream agent must pick the correct tool_id.

    If an explicit clue exists in the context, a base heuristic agent can solve it.
    Otherwise it requires (learned) memory guidance.
    """

    context_tokens: Tuple[str, ...]
    tool_id: int
    has_explicit_clue: bool


TOOL_NAMES = (
    "search",
    "filter_price",
    "filter_rating",
    "open_product",
)


class Vocab:
    def __init__(self, tokens: Iterable[str]):
        uniq = sorted(set(tokens))
        self.token_to_id: Dict[str, int] = {t: i for i, t in enumerate(uniq, start=1)}
        self.token_to_id["<pad>"] = 0
        self.id_to_token: Dict[int, str] = {i: t for t, i in self.token_to_id.items()}

    @property
    def pad_id(self) -> int:
        return 0

    def encode(self, tokens: Sequence[str]) -> List[int]:
        return [self.token_to_id[t] for t in tokens]


def build_vocab(num_tools: int) -> Vocab:
    keywords = [f"kw:{i}" for i in range(num_tools)]
    domains = ["domain:shopping", "domain:cms", "domain:reddit"]
    explicit = [f"explicit:tool_{i}" for i in range(num_tools)]
    misc = ["user:intent", "session:history", "obs:page"]
    return Vocab([*keywords, *domains, *explicit, *misc])


def generate_tasks(
    *,
    n: int,
    num_tools: int,
    explicit_ratio: float,
    seed: int,
) -> List[Task]:
    rng = np.random.default_rng(seed)
    tasks: List[Task] = []

    for _ in range(n):
        tool_id = int(rng.integers(0, num_tools))
        has_explicit = bool(rng.random() < explicit_ratio)

        domain = rng.choice(["domain:shopping", "domain:cms", "domain:reddit"]).item()
        context = [domain, "user:intent", "obs:page", f"kw:{tool_id}"]

        if has_explicit:
            context.append(f"explicit:tool_{tool_id}")
        # If there is no explicit clue, the downstream agent will fall back to a weak default.

        rng.shuffle(context)
        tasks.append(Task(tuple(context), tool_id=tool_id, has_explicit_clue=has_explicit))

    return tasks


def build_offline_experience_bank(tasks: Sequence[Task]) -> List[Tuple[Task, int]]:
    """Stage-1 experience bank.

    We only store guidance for tasks without explicit clues, since those are the ones
    where memory is actually useful.

    Guidance is represented as a discrete hint id == tool_id.
    """

    return [(t, t.tool_id) for t in tasks if not t.has_explicit_clue]
