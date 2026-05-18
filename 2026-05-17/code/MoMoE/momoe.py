"""
Full MoMoE Pipeline.

Implements the complete Allocate → Predict → Aggregate → Explain pipeline
from the paper "MoMoE: Mixture of Moderation Experts Framework for
AI-Assisted Online Governance" (EMNLP 2025).

Community labels (7 experts for MoMoE-Community):
  askreddit, relationships, gaming, worldnews, technology, science, fitness

Norm-violation labels (5 experts for MoMoE-NormVio):
  harassment, hate_speech, misinformation, spam_scam, explicit_content
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from models.allocator import Allocator, AllocationResult
from models.expert import ExpertPool, ExpertPrediction
from models.explainer import MoMoEExplainer
from operators.aggregate import aggregate_batch, AggregateResult

logger = logging.getLogger(__name__)

COMMUNITY_EXPERTS = [
    "askreddit", "relationships", "gaming", "worldnews",
    "technology", "science", "fitness"
]

NORM_VIOLATION_EXPERTS = [
    "harassment", "hate_speech", "misinformation",
    "spam_scam", "explicit_content"
]

COMMUNITY_NORMS = {
    "askreddit": ["No personal attacks", "Stay on topic", "No spam"],
    "relationships": ["Be supportive", "No harassment", "Respect privacy"],
    "gaming": ["No cheating/hacking discussions", "Be respectful", "No spoilers without tags"],
    "worldnews": ["No misinformation", "Cite sources", "Stay relevant to news"],
    "technology": ["No misleading technical claims", "No spam", "Be accurate"],
    "science": ["No pseudoscience", "Cite evidence", "Peer-reviewed sources preferred"],
    "fitness": ["No harmful advice", "Consult professionals", "No supplement spam"],
}


@dataclass
class MoMoEOutput:
    text: str
    label: int          # 0 = compliant, 1 = violation
    confidence: float
    community: str
    active_experts: list[str]
    expert_predictions: dict[str, float]  # expert_id -> violation confidence
    aggregation_strategy: str
    explanation: Optional[dict] = None


class MoMoE:
    """
    Full MoMoE inference pipeline.

    MoMoE-Community: 7 community-specialized experts
    MoMoE-NormVio:   5 norm-violation specialized experts

    Usage:
        momoe = MoMoE(mode="community", device="cuda")
        outputs = momoe.predict(["post text 1", "post text 2"])
    """

    def __init__(
        self,
        mode: str = "community",          # "community" or "norm_violation"
        top_k: int = 3,                   # Top-K experts to activate
        aggregation: str = "weighted",    # "weighted" or "majority_vote"
        device: str = "cpu",
        allocator_checkpoint: Optional[str] = None,
        expert_checkpoint_dir: Optional[str] = None,
        use_explainer: bool = False,
        openai_api_key: Optional[str] = None,
    ):
        self.mode = mode
        self.top_k = top_k
        self.aggregation = aggregation
        self.device = device

        expert_ids = COMMUNITY_EXPERTS if mode == "community" else NORM_VIOLATION_EXPERTS

        logger.info(f"Initializing MoMoE-{mode.title()} with {len(expert_ids)} experts")

        # [Step 1] Allocate: RoBERTa-base classifier
        self.allocator = Allocator(
            allocator_type=mode,
            checkpoint=allocator_checkpoint,
        )

        # [Step 2] Predict: specialized expert pool
        self.expert_pool = ExpertPool(
            expert_ids=expert_ids,
            checkpoint_dir=expert_checkpoint_dir,
            device=device,
        )

        # [Step 4] Explain: optional GPT-4o explainer
        self.explainer = None
        if use_explainer:
            self.explainer = MoMoEExplainer(api_key=openai_api_key)

    def predict(
        self,
        texts: list[str],
        generate_explanations: bool = False,
    ) -> list[MoMoEOutput]:
        """
        Run the full MoMoE pipeline for a batch of texts.

        Pipeline:
        1. Allocate: determine top-K active experts and their weights
        2. Predict: run each active expert on all texts
        3. Aggregate: combine expert predictions
        4. Explain: (optional) generate post-hoc explanations
        """

        # Step 1: Allocate
        allocations: list[AllocationResult] = self.allocator(
            texts, top_k=self.top_k, device=self.device
        )

        # Collect unique active experts across the batch
        all_active_experts: set[str] = set()
        for alloc in allocations:
            all_active_experts.update(alloc.expert_labels)

        # Step 2: Predict (batch inference for each active expert)
        expert_predictions: dict[str, list[ExpertPrediction]] = (
            self.expert_pool.predict(texts, list(all_active_experts))
        )

        # Step 3: Aggregate
        aggregate_results: list[AggregateResult] = aggregate_batch(
            expert_predictions, allocations, strategy=self.aggregation
        )

        # Step 4: Explain (optional)
        outputs = []
        for i, (text, alloc, agg) in enumerate(zip(texts, allocations, aggregate_results)):
            expert_conf = {
                eid: expert_predictions[eid][i].confidence
                for eid in alloc.expert_labels
                if eid in expert_predictions and i < len(expert_predictions[eid])
            }

            explanation = None
            if generate_explanations and self.explainer:
                norms = COMMUNITY_NORMS.get(alloc.expert_labels[0], []) if alloc.expert_labels else []
                explanation = self.explainer.explain(
                    post_text=text,
                    verdict=agg.label,
                    community=alloc.expert_labels[0] if alloc.expert_labels else "general",
                    active_norms=norms[:2],
                    expert_confidences=expert_conf,
                )

            outputs.append(MoMoEOutput(
                text=text,
                label=agg.label,
                confidence=agg.confidence,
                community=alloc.expert_labels[0] if alloc.expert_labels else "unknown",
                active_experts=alloc.expert_labels,
                expert_predictions=expert_conf,
                aggregation_strategy=self.aggregation,
                explanation=explanation,
            ))

        return outputs
