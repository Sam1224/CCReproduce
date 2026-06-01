"""
Controllable Reasoning Modes for Valley3 (toy implementation).

Paper (Section 3.2, Post-training):
Valley3 supports one non-thinking mode and three distinct thinking levels:
  - Level 0 (Non-thinking): direct generation, fast
  - Level 1: brief internal reasoning before answer
  - Level 2: moderate chain-of-thought reasoning
  - Level 3: deep, long-chain reasoning (for complex compliance/analysis tasks)

The thinking level is controlled via the <think> token budget.
In generation: the model produces <think>...</think><answer>...</answer>

This module implements the generation logic with thinking level control.
"""
import torch
from dataclasses import dataclass
from typing import Optional

THINK_START = "<think>"
THINK_END = "</think>"
ANSWER_START = "<answer>"
ANSWER_END = "</answer>"

# Max thinking tokens per level (toy values; real Valley3 uses GRPO-learned budgets)
THINK_BUDGET = {
    0: 0,    # Non-thinking: no internal reasoning
    1: 32,   # Brief reasoning
    2: 128,  # Moderate reasoning
    3: 512,  # Deep reasoning (for violation analysis, compliance checks)
}


@dataclass
class ThinkingConfig:
    level: int = 0       # 0-3
    temperature: float = 0.8
    max_new_tokens: int = 256

    @property
    def max_think_tokens(self) -> int:
        return THINK_BUDGET[self.level]


def format_prompt_with_thinking(question: str, think_level: int) -> str:
    """
    Format the input prompt to indicate desired thinking mode.
    Paper: controllable reasoning is requested via special tokens / instruction.
    """
    if think_level == 0:
        return f"<|user|>{question}<|assistant|>{ANSWER_START}"
    else:
        return (f"<|user|>{question}<|think_level|>{think_level}<|assistant|>"
                f"{THINK_START}")


def parse_thinking_output(output: str, think_level: int) -> dict:
    """Parse model output into thinking trace + answer."""
    if think_level == 0:
        return {"thinking": None, "answer": output.strip()}
    if THINK_END in output:
        parts = output.split(THINK_END, 1)
        thinking = parts[0].replace(THINK_START, "").strip()
        answer = parts[1].replace(ANSWER_START, "").replace(ANSWER_END, "").strip()
    else:
        thinking = output
        answer = ""
    return {"thinking": thinking, "answer": answer}


class ControllableReasoner:
    """
    Wraps a generative model with thinking mode control.
    In toy mode, simulates different verbosity levels.
    """
    def __init__(self, model=None, config: Optional[ThinkingConfig] = None):
        self.model = model  # real model; toy uses template responses
        self.config = config or ThinkingConfig()

    def generate(self, question: str, context: str = "",
                 think_level: Optional[int] = None) -> dict:
        level = think_level if think_level is not None else self.config.level
        budget = THINK_BUDGET[level]

        # Toy simulation: generate template responses at different verbosity
        if level == 0:
            answer = f"[Direct] {question[:50]}... → Classification: OK"
            return {"thinking": None, "answer": answer, "think_tokens": 0}
        elif level == 1:
            thinking = f"Brief check: Is content violating policy? Checking key claims..."
            answer = f"[L1] Result: Minor concern detected."
            return {"thinking": thinking, "answer": answer, "think_tokens": len(thinking.split())}
        elif level == 2:
            thinking = (f"Step 1: Identify claims in content.\n"
                       f"Step 2: Map to relevant policy rules.\n"
                       f"Step 3: Evaluate evidence strength...")
            answer = f"[L2] Violation detected: possible false efficacy claim."
            return {"thinking": thinking, "answer": answer, "think_tokens": len(thinking.split())}
        else:  # level == 3
            thinking = (f"Deep analysis — Step 1: Parse all product claims.\n"
                       f"Step 2: Cross-reference with advertising law database.\n"
                       f"Step 3: Evaluate each claim against Rule 3.2.1 (efficacy).\n"
                       f"Step 4: Check for exaggeration patterns.\n"
                       f"Step 5: Assess context and disclaimer presence.\n"
                       f"Conclusion: Clear violation of Rule 3.2.1 with high confidence.")
            answer = f"[L3] HIGH CONFIDENCE VIOLATION: False efficacy claim, Rule 3.2.1."
            return {"thinking": thinking, "answer": answer, "think_tokens": len(thinking.split())}
