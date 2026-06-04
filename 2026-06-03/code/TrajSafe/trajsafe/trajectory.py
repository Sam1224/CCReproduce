"""
Conversation Trajectory Representation for TrajSafe.

A trajectory is a sequence of (user, assistant) turns.
TrajSafe monitors the trajectory at each turn and decides whether to intervene.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class HarmLevel(Enum):
    SAFE = 0
    BORDERLINE = 1
    HARMFUL = 2
    SEVERELY_HARMFUL = 3


@dataclass
class Turn:
    role: str        # "user" or "assistant"
    content: str
    harm_level: HarmLevel = HarmLevel.SAFE
    # TrajSafe annotations
    is_amplification_step: bool = False
    amplification_score: float = 0.0  # [0, 1] how much this turn amplifies harm


@dataclass
class Trajectory:
    """
    Multi-turn conversation trajectory.
    TrajSafe processes this causally: after each turn, decides whether to intervene.
    """
    scenario_id: str
    category: str
    turns: List[Turn] = field(default_factory=list)
    # Ground truth annotation
    harm_amplified: bool = False          # Does the full trajectory amplify harm?
    amplification_turn_idx: Optional[int] = None  # Which turn unlocks harm capability

    def add_turn(self, role: str, content: str) -> None:
        self.turns.append(Turn(role=role, content=content))

    def get_context_window(self, max_turns: int = 10) -> List[Turn]:
        """Get recent conversation context."""
        return self.turns[-max_turns:]

    def to_text(self) -> str:
        """Format trajectory as text."""
        lines = []
        for t in self.turns:
            lines.append(f"[{t.role.upper()}]: {t.content}")
        return "\n".join(lines)

    @property
    def num_turns(self) -> int:
        return len(self.turns)
