"""Recall agent: identifies videos potentially containing emerging issues.

In the paper, a multimodal LLM Agent analyzes video content (visuals + audio + OCR)
and compares against the existing policy library to identify candidates for
emerging issue discovery.

This toy implementation simulates the agent using rule-based heuristics
(since we don't have a real LLM endpoint in this environment).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from data.synthetic_videos import SyntheticVideo, EXISTING_POLICIES
from agent.prompts import format_recall_prompt


@dataclass
class RecallResult:
    video_id: str
    recalled: bool
    reason: str
    potential_issue: str = ""


def _heuristic_recall(video: SyntheticVideo) -> RecallResult:
    """Heuristic simulation of LLM recall agent.

    Real implementation would call an LLM API with the formatted prompts.
    """
    desc_lower = video.description.lower()
    tags_lower = [t.lower() for t in video.visual_tags]

    # Skip if existing policy clearly covers it
    for policy in EXISTING_POLICIES:
        if policy in desc_lower:
            return RecallResult(
                video_id=video.video_id,
                recalled=False,
                reason=f"Covered by existing policy: {policy}",
            )

    # Recall heuristics for emerging issues
    emerging_signals = {
        "financial": ["investment", "returns", "crypto", "lottery", "wealth", "membership"],
        "tobacco": ["smoking", "tobacco", "lifestyle accessories", "club"],
        "health_misleading": ["cure", "dissolves fat", "miracle", "herb", "supplement"],
        "subtle": ["subtle", "variant"],
    }

    for issue_type, signals in emerging_signals.items():
        if any(s in desc_lower or s in tags_lower for s in signals):
            return RecallResult(
                video_id=video.video_id,
                recalled=True,
                reason=f"Potential emerging issue: {issue_type}",
                potential_issue=f"Possibly related to {issue_type} content not in current policy",
            )

    # Benign content
    return RecallResult(
        video_id=video.video_id,
        recalled=False,
        reason="Content appears benign and covered by normal behavior",
    )


class RecallAgent:
    """LLM-based recall agent for emerging content issue discovery.

    Paper §3.1: The agent analyzes multimodal video content and identifies
    candidates that may contain emerging issues outside the current policy scope.
    """

    def __init__(self, use_llm: bool = False, llm_model: str = "gpt-4o"):
        self.use_llm = use_llm
        self.llm_model = llm_model

    def recall(self, video: SyntheticVideo) -> RecallResult:
        if self.use_llm:
            return self._llm_recall(video)
        return _heuristic_recall(video)

    def _llm_recall(self, video: SyntheticVideo) -> RecallResult:
        """Real LLM-based recall (requires API key)."""
        # In production: call OpenAI / local LLM with formatted prompts
        # system, user = format_recall_prompt(video.description, video.visual_tags)
        # response = openai.chat.completions.create(...)
        # parse response for RECALL: YES/NO
        raise NotImplementedError(
            "LLM recall requires API key. Set use_llm=False for heuristic mode."
        )

    def batch_recall(
        self,
        videos: list[SyntheticVideo],
        verbose: bool = False,
    ) -> list[RecallResult]:
        results = []
        for i, video in enumerate(videos):
            result = self.recall(video)
            results.append(result)
            if verbose and i % 50 == 0:
                n_recalled = sum(r.recalled for r in results)
                print(f"[RecallAgent] {i}/{len(videos)} processed, {n_recalled} recalled")
        return results
