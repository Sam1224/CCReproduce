"""
Explain Operator for MoMoE.

Generates post-hoc explanations for moderation decisions.

Paper: "The Explain operator generates concise, post-hoc justifications
for MoMoE's decision in a three-level JSON format using a large language
model like GPT-4o."

Three-level explanation format:
{
  "level_1": "Brief verdict (1 sentence)",
  "level_2": "Rule identification (which norm was violated)",
  "level_3": "Detailed reasoning with evidence from the post"
}
"""

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

EXPLAIN_SYSTEM_PROMPT = """You are an AI content moderation assistant.
Your task is to explain content moderation decisions in a structured format.
Always respond with valid JSON only."""

EXPLAIN_PROMPT_TEMPLATE = """A content moderation system has classified the following post as: {verdict}.

Post: {post_text}
Community/Context: {community}
Relevant norm(s): {norms}

Please provide a concise explanation in the following JSON format:
{{
  "level_1": "One-sentence verdict summary",
  "level_2": "Which specific norm/rule was violated (or why it's compliant)",
  "level_3": "Detailed reasoning with specific evidence from the post (2-3 sentences)"
}}

Respond ONLY with the JSON object, nothing else."""


class MoMoEExplainer:
    """
    GPT-4o (or local LLM) based explainer for MoMoE decisions.

    The explainer is invoked after the Aggregate step to provide
    human-readable justifications. It is post-hoc and does not
    influence the moderation decision.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        use_local: bool = False,
    ):
        self.model = model
        self.use_local = use_local

        if not use_local:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=api_key)
            except ImportError:
                logger.warning("openai not installed; using dummy explainer")
                self.client = None

    def explain(
        self,
        post_text: str,
        verdict: int,           # 0 = compliant, 1 = violation
        community: str,
        active_norms: list[str],
        expert_confidences: dict[str, float],
    ) -> dict:
        """
        Generate three-level JSON explanation for a moderation decision.

        Args:
            post_text: The content being moderated
            verdict: Final binary label from Aggregate step
            community: Community/context label
            active_norms: Relevant norm violation categories
            expert_confidences: Dict of expert_id -> confidence
        """
        verdict_str = "VIOLATION" if verdict == 1 else "COMPLIANT"
        norms_str = ", ".join(active_norms) if active_norms else "general community guidelines"

        prompt = EXPLAIN_PROMPT_TEMPLATE.format(
            verdict=verdict_str,
            post_text=post_text[:500],  # Truncate for API efficiency
            community=community,
            norms=norms_str,
        )

        if self.use_local or self.client is None:
            return self._dummy_explanation(verdict, community, active_norms)

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": EXPLAIN_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=300,
                temperature=0.3,
                response_format={"type": "json_object"},
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            logger.warning(f"Explainer API call failed: {e}")
            return self._dummy_explanation(verdict, community, active_norms)

    def _dummy_explanation(
        self, verdict: int, community: str, active_norms: list[str]
    ) -> dict:
        """Fallback explanation without API access."""
        verdict_str = "violates community guidelines" if verdict == 1 else "complies with community guidelines"
        norms_str = " and ".join(active_norms) if active_norms else "general guidelines"
        return {
            "level_1": f"This post {verdict_str}.",
            "level_2": f"Relevant norm: {norms_str}",
            "level_3": (
                f"Based on expert analysis of {community} community standards, "
                f"the post was classified as {'violating' if verdict == 1 else 'complying with'} "
                f"the norms related to {norms_str}."
            ),
        }
