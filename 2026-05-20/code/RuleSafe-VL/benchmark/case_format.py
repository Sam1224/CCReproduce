"""
RuleSafe-VL Case Format
=======================
Defines the schema for benchmark cases.

Each case in the 2,166-case benchmark consists of:
  - image: PIL image or image path
  - text_context: accompanying text (caption, comment, description)
  - applicable_rules: list of rule IDs from the rule graph that are relevant
  - violated_rules: ground-truth violated rule IDs (may be empty = compliant)
  - moderation_outcome: ALLOW | RESTRICT | REMOVE
  - reasoning_chain: gold reasoning steps (for chain accuracy evaluation)

Evaluation tasks (per paper §4):
  Task A: Rule Retrieval — which rules are applicable?
  Task B: Rule Application — for applicable rules, which are violated?
  Task C: Moderation Decision — final ALLOW/RESTRICT/REMOVE outcome
  Task D: Full Chain — correct A + B + C
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
import json
from pathlib import Path


class ModerationOutcome(str, Enum):
    ALLOW = "ALLOW"
    RESTRICT = "RESTRICT"   # age-gate, content warning, limited distribution
    REMOVE = "REMOVE"


@dataclass
class ReasoningStep:
    step_id: int
    rule_id: str
    condition_check: str    # description of condition evaluation
    outcome: str            # "activated" | "overridden" | "exception_applied"
    evidence: str           # image/text cues supporting this step


@dataclass
class BenchmarkCase:
    case_id: str
    image_path: Optional[str]           # path to image file or None for text-only
    text_context: str                   # accompanying text content
    platform_context: str               # e.g., "general_feed", "adult_platform"
    applicable_rules: List[str]         # Task A ground truth
    violated_rules: List[str]           # Task B ground truth (subset of applicable)
    moderation_outcome: ModerationOutcome  # Task C ground truth
    reasoning_chain: List[ReasoningStep]   # Task D ground truth
    policy_family: str                  # F1 | F2 | F3
    difficulty: str = "medium"          # easy | medium | hard (based on rule interaction complexity)
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "image_path": self.image_path,
            "text_context": self.text_context,
            "platform_context": self.platform_context,
            "applicable_rules": self.applicable_rules,
            "violated_rules": self.violated_rules,
            "moderation_outcome": self.moderation_outcome.value,
            "policy_family": self.policy_family,
            "difficulty": self.difficulty,
            "reasoning_chain": [
                {"step_id": s.step_id, "rule_id": s.rule_id,
                 "condition_check": s.condition_check, "outcome": s.outcome,
                 "evidence": s.evidence}
                for s in self.reasoning_chain
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BenchmarkCase":
        chain = [ReasoningStep(**s) for s in d.get("reasoning_chain", [])]
        return cls(
            case_id=d["case_id"],
            image_path=d.get("image_path"),
            text_context=d["text_context"],
            platform_context=d.get("platform_context", "general_feed"),
            applicable_rules=d["applicable_rules"],
            violated_rules=d["violated_rules"],
            moderation_outcome=ModerationOutcome(d["moderation_outcome"]),
            reasoning_chain=chain,
            policy_family=d.get("policy_family", "F1"),
            difficulty=d.get("difficulty", "medium"),
            notes=d.get("notes", ""),
        )


def load_cases(path: str) -> List[BenchmarkCase]:
    """Load cases from a JSONL file."""
    cases = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(BenchmarkCase.from_dict(json.loads(line)))
    return cases


def save_cases(cases: List[BenchmarkCase], path: str):
    with open(path, "w") as f:
        for case in cases:
            f.write(json.dumps(case.to_dict()) + "\n")
