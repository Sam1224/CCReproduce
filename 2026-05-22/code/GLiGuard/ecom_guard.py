"""
GLiGuard E-Commerce Content Governance Adaptation
=================================================
Adapts GLiGuard (arXiv:2605.07982) for influencer (达人) content moderation
on e-commerce live-streaming and short-video platforms.

Official model: fastino/gliguard-LLMGuardrails-300M
Official code:  https://github.com/fastino-ai/GLiGuard

Schema-conditioned classification approach:
  - Task names + candidate labels are encoded as structured token schema
  - Single bidirectional encoder pass covers all moderation dimensions
  - 0.3B parameters, 16x throughput vs 7B-27B decoder-based guards
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import json


# ---------------------------------------------------------------------------
# E-Commerce content policy schema
# ---------------------------------------------------------------------------

ECOM_POLICY_SCHEMA = {
    "task_names": [
        "prompt_safety",        # Is the text safe to show to users?
        "response_safety",      # Is the model response compliant?
        "harm_categories",      # Fine-grained harm classification
        "ad_compliance",        # Advertising law compliance (广告法)
        "health_claim",         # Unverified health / slimming claims
        "counterfeit_signal",   # Counterfeit or IP-infringing content
        "refusal_detection",    # Did the model appropriately refuse?
    ],
    "candidate_labels": [
        # Safety
        "safe", "unsafe",
        # Harm categories (ecom-specific)
        "misleading_ad",
        "prohibited_health_claim",   # 保健品夸大宣传
        "prohibited_weight_loss",    # 减肥产品违禁词
        "counterfeit_product",
        "price_manipulation",
        "false_celebrity_endorsement",
        "undisclosed_sponsored_content",
        # Policy compliance
        "ad_law_compliant",
        "ad_law_violation",          # 广告法违规 (绝对化用语等)
        "refusal",
        "not_refusal",
    ],
}


@dataclass
class ModerationResult:
    """Structured output for a single content moderation call."""
    text: str
    is_safe: bool
    harm_labels: list[str] = field(default_factory=list)
    ad_compliant: bool = True
    health_claim_detected: bool = False
    counterfeit_signal: bool = False
    raw_scores: dict = field(default_factory=dict)
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "text": self.text[:120] + "..." if len(self.text) > 120 else self.text,
            "is_safe": self.is_safe,
            "harm_labels": self.harm_labels,
            "ad_compliant": self.ad_compliant,
            "health_claim_detected": self.health_claim_detected,
            "counterfeit_signal": self.counterfeit_signal,
            "confidence": round(self.confidence, 3),
        }


class EComGuard:
    """
    Thin wrapper around GLiGuard for e-commerce content governance.

    Usage:
        guard = EComGuard()
        result = guard.moderate("买这个减肥茶，7天瘦10斤，效果绝对保证！")
        print(result.is_safe, result.harm_labels)

    Note: Requires `gliner` package (GLiNER2 interface).
      pip install gliner2 transformers torch
    """

    MODEL_ID = "fastino/gliguard-LLMGuardrails-300M"

    def __init__(self, model_id: str = MODEL_ID, device: str = "cpu"):
        self.model_id = model_id
        self.device = device
        self._model = None  # lazy load

    def _load_model(self):
        """Lazy-load GLiGuard model on first use."""
        # Requires: pip install gliner2
        try:
            from gliner import GLiNER  # GLiNER2 interface
            self._model = GLiNER.from_pretrained(self.model_id)
            if self.device != "cpu":
                self._model = self._model.to(self.device)
        except ImportError:
            raise ImportError(
                "GLiNER2 not installed. Run: pip install gliner2 transformers torch"
            )

    def moderate(self, text: str, threshold: float = 0.5) -> ModerationResult:
        """
        Moderate a single text using GLiGuard schema-conditioned classification.

        Args:
            text: Input text to moderate (influencer script, product description, etc.)
            threshold: Classification confidence threshold

        Returns:
            ModerationResult with structured safety assessment
        """
        if self._model is None:
            self._load_model()

        raw = self._model.classify_text(
            text=text,
            task_names=ECOM_POLICY_SCHEMA["task_names"],
            candidate_labels=ECOM_POLICY_SCHEMA["candidate_labels"],
        )

        # Parse structured output
        # (exact API depends on GLiNER2 version; adapt as needed)
        harm_labels = []
        is_safe = True
        ad_compliant = True
        health_claim = False
        counterfeit = False
        confidence = 0.0

        if isinstance(raw, dict):
            # Example parsing — adjust to actual GLiNER2 output format
            for key, score in raw.items():
                if score > threshold:
                    if key in ("unsafe", "misleading_ad", "prohibited_health_claim",
                               "prohibited_weight_loss", "false_celebrity_endorsement",
                               "undisclosed_sponsored_content", "price_manipulation"):
                        harm_labels.append(key)
                        is_safe = False
                    if key in ("ad_law_violation",):
                        ad_compliant = False
                    if key in ("prohibited_health_claim", "prohibited_weight_loss"):
                        health_claim = True
                    if key == "counterfeit_product":
                        counterfeit = True
                    confidence = max(confidence, score)

        return ModerationResult(
            text=text,
            is_safe=is_safe,
            harm_labels=harm_labels,
            ad_compliant=ad_compliant,
            health_claim_detected=health_claim,
            counterfeit_signal=counterfeit,
            raw_scores=raw if isinstance(raw, dict) else {},
            confidence=confidence,
        )

    def moderate_batch(self, texts: list[str], threshold: float = 0.5) -> list[ModerationResult]:
        """Moderate a batch of texts."""
        return [self.moderate(t, threshold) for t in texts]


# ---------------------------------------------------------------------------
# Demo / toy run (no model required — shows expected I/O)
# ---------------------------------------------------------------------------

DEMO_TEXTS = [
    # Violating: prohibited health claim + absolute superlative (广告法违规)
    "买这个减肥茶，7天瘦10斤，效果绝对保证！国内第一减肥神器！",
    # Violating: unverified medical claim
    "本品可根治糖尿病，三个疗程保证痊愈，真实案例分享！",
    # Compliant: standard product description
    "这款面霜质地清爽，适合夏季使用，请根据个人肤质选购。",
    # Borderline: sponsored but disclosed
    "【广告】合作品牌赞助，我来体验一下这款新出的护肤品，大家看效果。",
    # Counterfeit signal
    "高仿LV包包，正品面料，专柜同款，无法区分真假！",
]

DEMO_EXPECTED = [
    {"is_safe": False, "harm": ["prohibited_weight_loss", "ad_law_violation"]},
    {"is_safe": False, "harm": ["prohibited_health_claim"]},
    {"is_safe": True, "harm": []},
    {"is_safe": True, "harm": []},  # disclosed, so compliant
    {"is_safe": False, "harm": ["counterfeit_product"]},
]


def demo_without_model():
    """Show expected input/output format without loading the actual model."""
    print("=" * 60)
    print("GLiGuard E-Commerce Moderation — Demo (no model loaded)")
    print("=" * 60)
    for text, expected in zip(DEMO_TEXTS, DEMO_EXPECTED):
        print(f"\nInput:    {text[:80]}...")
        print(f"Expected: is_safe={expected['is_safe']}, harm={expected['harm']}")
    print("\n[To run with actual model, call: guard = EComGuard(); guard.moderate(text)]")


if __name__ == "__main__":
    demo_without_model()
