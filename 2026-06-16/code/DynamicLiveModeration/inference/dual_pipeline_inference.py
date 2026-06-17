"""
Dual-pipeline inference for production livestream moderation.

Paper (§3): The two pipelines run in parallel; violations detected by either
pipeline are flagged. This "OR-gate" union design maximizes recall.

Pipeline A (Classification): Fast, optimized for known violation types.
Pipeline B (Similarity): Catches novel/subtle violations via reference matching.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

from models.mllm_teacher import TextEncoder, simple_tokenize
from models.classification_pipeline import ClassificationStudent
from models.similarity_pipeline import SimilarityStudent, ReferenceBank, SimilarityDecider


@dataclass
class ModerationDecision:
    segment_id: str
    is_violation: bool
    source: str          # "classification", "similarity", "both", "safe"
    category: str
    classification_score: float
    similarity_score: float
    confidence: float


class DualPipelineModerator:
    """
    Production dual-pipeline content moderator.

    Paper: Combines supervised classification (known violations) with
    reference-based similarity matching (novel violations).
    Both pipelines boosted by MLLM knowledge distillation.
    """

    def __init__(
        self,
        classifier: ClassificationStudent,
        similarity_model: SimilarityStudent,
        text_encoder: TextEncoder,
        reference_bank: ReferenceBank,
        cls_threshold: float = 0.5,
        sim_threshold: float = 0.7,
        device: str = "cpu",
    ):
        self.classifier = classifier.to(device)
        self.similarity_model = similarity_model.to(device)
        self.text_encoder = text_encoder.to(device)
        self.reference_bank = reference_bank
        self.sim_decider = SimilarityDecider(threshold=sim_threshold)
        self.cls_threshold = cls_threshold
        self.device = device

        self.classifier.eval()
        self.similarity_model.eval()
        self.text_encoder.eval()

    @torch.no_grad()
    def moderate_segment(
        self,
        segment_id: str,
        text: str,
        audio_feat: torch.Tensor,
        visual_feat: torch.Tensor,
    ) -> ModerationDecision:
        """
        Moderate a single livestream segment.

        Args:
            segment_id: unique identifier
            text: ASR transcript
            audio_feat: (audio_dim,) audio features
            visual_feat: (visual_dim,) visual features

        Returns:
            ModerationDecision with violation status and source
        """
        # Prepare inputs
        token_ids = simple_tokenize([text]).to(self.device)
        text_feat = self.text_encoder(token_ids)
        audio_feat = audio_feat.unsqueeze(0).to(self.device)
        visual_feat = visual_feat.unsqueeze(0).to(self.device)

        # Pipeline A: Supervised Classification
        cls_logits = self.classifier(text_feat, audio_feat, visual_feat)
        cls_prob = torch.softmax(cls_logits, dim=-1)[0, 1].item()
        cls_violation = cls_prob > self.cls_threshold

        # Pipeline B: Similarity Matching
        sim_embed = self.similarity_model(text_feat, audio_feat, visual_feat).squeeze(0)
        all_refs = self.reference_bank.get_all_refs().to(self.device)

        if len(all_refs) > 0:
            sims = torch.mv(all_refs, sim_embed)
            top_sims, top_idx = sims.topk(min(5, len(sims)))

            all_cats = []
            for cat, embeds in self.reference_bank.bank.items():
                all_cats.extend([cat] * len(embeds))
            top_cats = [all_cats[i] for i in top_idx.tolist()]
        else:
            top_sims = torch.zeros(0)
            top_cats = []

        sim_violation, sim_category, sim_score = self.sim_decider.decide(top_sims, top_cats)

        # Union decision: flag if either pipeline detects violation
        is_violation = cls_violation or sim_violation
        if cls_violation and sim_violation:
            source = "both"
            category = sim_category
        elif cls_violation:
            source = "classification"
            category = "unknown_violation"
        elif sim_violation:
            source = "similarity"
            category = sim_category
        else:
            source = "safe"
            category = "safe"

        confidence = max(cls_prob, sim_score)

        return ModerationDecision(
            segment_id=segment_id,
            is_violation=is_violation,
            source=source,
            category=category,
            classification_score=cls_prob,
            similarity_score=sim_score,
            confidence=confidence,
        )

    def moderate_batch(self, segments: List[Dict]) -> List[ModerationDecision]:
        """Moderate a batch of segments."""
        decisions = []
        for seg in segments:
            decision = self.moderate_segment(
                segment_id=seg["segment_id"],
                text=seg["text"],
                audio_feat=seg["audio_feat"],
                visual_feat=seg["visual_feat"],
            )
            decisions.append(decision)
        return decisions


def demo():
    """Demo: run dual pipeline on a few toy segments."""
    from data.toy_dataset import ToyLivestreamDataset

    device = "cpu"
    ds = ToyLivestreamDataset(num_samples=20, seed=99)

    text_enc = TextEncoder(vocab_size=1000, embed_dim=64, out_dim=128)
    cls_model = ClassificationStudent(audio_dim=512, visual_dim=768, text_dim=128,
                                       hidden_dim=256, num_classes=2)
    sim_model = SimilarityStudent(audio_dim=512, visual_dim=768, text_dim=128,
                                   hidden_dim=256, embed_dim=512)
    bank = ds.get_reference_bank()
    # Convert reference bank to ReferenceBank object
    ref_bank_obj = ReferenceBank(embed_dim=512)
    for cat, vecs in bank.items():
        for v in vecs:
            ref_bank_obj.add(cat, v)
    ref_bank_obj.build()

    moderator = DualPipelineModerator(
        classifier=cls_model,
        similarity_model=sim_model,
        text_encoder=text_enc,
        reference_bank=ref_bank_obj,
        cls_threshold=0.5,
        sim_threshold=0.7,
        device=device,
    )

    print("=== Dual Pipeline Inference Demo ===")
    for i, seg in enumerate(ds.samples[:5]):
        decision = moderator.moderate_segment(
            segment_id=seg.segment_id,
            text=seg.text,
            audio_feat=seg.audio_feat,
            visual_feat=seg.visual_feat,
        )
        gt = "VIOLATION" if seg.label == 1 else "SAFE"
        pred = "VIOLATION" if decision.is_violation else "SAFE"
        print(f"  [{i}] GT={gt:9s} Pred={pred:9s} "
              f"Source={decision.source:14s} "
              f"Cls={decision.classification_score:.3f} "
              f"Sim={decision.similarity_score:.3f}")

    print("\nPaper production results:")
    print("  Classification: 67% Recall @ 80% Precision")
    print("  Similarity:     76% Recall @ 80% Precision")
    print("  A/B test:       -6~8% user views of unwanted livestreams")


if __name__ == "__main__":
    demo()
