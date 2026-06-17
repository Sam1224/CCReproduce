"""
Three-Stage UNIVID Pipeline for Moderation.

Paper (§3): Full pipeline combining:
  Stage 1 – Risk Filter: pre-screens all segments, routes high-risk to actor
  Stage 2 – Moderation Actor:
      UNIVID-Lite (fast path for standard violations)
      UNIVID-RAG  (retrieval path for novel/uncertain violations)
  Stage 3 – Trend Governance: offline analysis, adaptive head deployment

Decision flow:
  segment
    → Risk Filter (risk_score)
    → [low risk: SAFE] | [high risk: Moderation Actor]
        → UNIVID-Lite → [confident: decision] | [uncertain: UNIVID-RAG]
            → UNIVID-RAG → final decision
    → check active Trend Heads
    → cache embedding for Trend Governance
"""

import torch
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List

from models.risk_filter import RiskFilter, TextFeatureExtractor, simple_tokenize
from models.policy_caption import PolicyCaptionEncoder, PolicyCaptionDecoder, PolicyAlignedEmbedding
from models.univid_lite import UniVIDLite
from models.univid_rag import UniVIDRAG, CaseLibrary
from models.trend_governance import TrendGovernanceSystem, CachedEmbedding


CATEGORY_NAMES = [
    "safe",
    "explicit_content",
    "hate_speech",
    "violence",
    "spam_promotion",
    "misinformation",
    "copyright",
    "underage",
]


@dataclass
class ModerationResult:
    segment_id: str
    is_violation: bool
    category: str
    confidence: float
    stage: str                # "filter_safe" | "lite" | "rag" | "trend"
    risk_score: float
    latency_ms: float
    lite_uncertainty: float = 0.0
    retrieved_cases: int = 0


class UniVIDPipeline:
    """
    Full three-stage UNIVID content moderation pipeline.

    Instantiate once, call moderate_segment() per content item.
    """

    def __init__(
        self,
        text_extractor: TextFeatureExtractor,
        risk_filter: RiskFilter,
        caption_encoder: PolicyCaptionEncoder,
        caption_decoder: PolicyCaptionDecoder,
        univid_lite: UniVIDLite,
        univid_rag: UniVIDRAG,
        case_library: CaseLibrary,
        trend_system: TrendGovernanceSystem,
        # Routing thresholds
        filter_threshold: float = 0.3,
        uncertainty_threshold: float = 0.5,
        rag_threshold: float = 0.5,
        trend_threshold: float = 0.5,
        device: str = "cpu",
    ):
        self.text_ext = text_extractor.to(device)
        self.risk_filter = risk_filter.to(device)
        self.caption_encoder = caption_encoder.to(device)
        self.caption_decoder = caption_decoder.to(device)
        self.univid_lite = univid_lite.to(device)
        self.univid_rag = univid_rag.to(device)
        self.case_library = case_library
        self.trend_system = trend_system
        self.filter_threshold = filter_threshold
        self.uncertainty_threshold = uncertainty_threshold
        self.rag_threshold = rag_threshold
        self.trend_threshold = trend_threshold
        self.device = device

        # Set all models to eval
        for m in [self.text_ext, self.risk_filter, self.caption_encoder,
                  self.caption_decoder, self.univid_lite, self.univid_rag]:
            m.eval()

    @torch.no_grad()
    def moderate_segment(
        self,
        segment_id: str,
        text: str,
        visual_feat: torch.Tensor,
        timestamp: float = 0.0,
    ) -> ModerationResult:
        t0 = time.perf_counter()

        visual_feat = visual_feat.unsqueeze(0).to(self.device)  # (1, V)

        # ── Stage 1: Risk Filter ──────────────────────────────────────
        token_ids = simple_tokenize([text]).to(self.device)
        text_feat = self.text_ext(token_ids)                     # (1, T)
        risk_score = self.risk_filter(visual_feat, text_feat)[0].item()

        if risk_score <= self.filter_threshold:
            latency = (time.perf_counter() - t0) * 1000
            return ModerationResult(
                segment_id=segment_id,
                is_violation=False,
                category="safe",
                confidence=1.0 - risk_score,
                stage="filter_safe",
                risk_score=risk_score,
                latency_ms=latency,
            )

        # ── Stage 2a: Policy Caption Generation ──────────────────────
        latent = self.caption_encoder(visual_feat, text_feat)    # (1, L)
        caption_embed = self.caption_decoder(latent)              # (1, C)

        # ── Stage 2b: UNIVID-Lite ──────────────────────────────────────
        binary_logits, cat_logits, feat = self.univid_lite(visual_feat, caption_embed)
        uncertainty = self.univid_lite.uncertainty(cat_logits)[0].item()

        if uncertainty <= self.uncertainty_threshold:
            # Confident decision from Lite
            cat_idx = cat_logits.argmax(dim=-1)[0].item()
            category = CATEGORY_NAMES[cat_idx]
            is_violation = binary_logits.argmax(dim=-1)[0].item() == 1
            confidence = torch.softmax(binary_logits, dim=-1)[0, 1].item()
            stage = "lite"
        else:
            # ── Stage 2c: UNIVID-RAG ──────────────────────────────────
            query_embed = self.univid_rag.query_encoder(visual_feat, caption_embed)[0]
            retrieved = self.case_library.retrieve(query_embed, top_k=5, device=self.device)

            if retrieved:
                ret_embeds = torch.stack([r.embed for r in retrieved]).unsqueeze(0).to(self.device)
            else:
                ret_embeds = torch.zeros(1, 1, self.univid_rag.embed_dim, device=self.device)

            _, bin_rag, cat_rag = self.univid_rag(visual_feat, caption_embed, ret_embeds)
            cat_idx = cat_rag.argmax(dim=-1)[0].item()
            category = CATEGORY_NAMES[cat_idx]
            is_violation = bin_rag.argmax(dim=-1)[0].item() == 1
            confidence = torch.softmax(bin_rag, dim=-1)[0, 1].item()
            stage = "rag"
            uncertainty_out = uncertainty

        # ── Stage 3: Check Trend Heads ────────────────────────────────
        trend_hit = self.trend_system.check_trend_heads(
            feat.squeeze(0), threshold=self.trend_threshold
        )
        if trend_hit is not None:
            is_violation = True
            category = trend_hit
            stage = "trend"
            confidence = max(confidence, 0.8)

        # Cache embedding for offline trend analysis
        self.trend_system.ingest(CachedEmbedding(
            segment_id=segment_id,
            embed=feat.squeeze(0).cpu(),
            decision=category,
            timestamp=timestamp,
            is_violation=is_violation,
        ))

        latency = (time.perf_counter() - t0) * 1000
        return ModerationResult(
            segment_id=segment_id,
            is_violation=is_violation,
            category=category,
            confidence=confidence,
            stage=stage,
            risk_score=risk_score,
            latency_ms=latency,
            lite_uncertainty=uncertainty,
            retrieved_cases=len(retrieved) if stage == "rag" else 0,
        )

    def moderate_batch(
        self,
        segments: List[Dict],
    ) -> List[ModerationResult]:
        return [
            self.moderate_segment(
                segment_id=s["segment_id"],
                text=s["text"],
                visual_feat=s["visual_feat"],
                timestamp=s.get("timestamp", 0.0),
            )
            for s in segments
        ]


def build_default_pipeline(device: str = "cpu") -> UniVIDPipeline:
    """Build a pipeline with default hyperparameters for toy experiments."""
    from models.trend_governance import TrendGovernanceSystem

    text_ext = TextFeatureExtractor(vocab_size=10000, embed_dim=128, out_dim=256)
    risk_filter = RiskFilter(visual_dim=768, text_dim=256, hidden_dim=256)
    cap_encoder = PolicyCaptionEncoder(visual_dim=768, text_dim=256, latent_dim=512)
    cap_decoder = PolicyCaptionDecoder(latent_dim=512, caption_dim=512)
    lite = UniVIDLite(visual_dim=768, caption_dim=512, hidden_dim=512,
                      num_categories=8, nhead=8, num_layers=3)
    rag = UniVIDRAG(visual_dim=768, caption_dim=512, embed_dim=512,
                    num_categories=8, top_k=5)
    case_lib = CaseLibrary(embed_dim=512)
    trend_sys = TrendGovernanceSystem(embed_dim=512)

    return UniVIDPipeline(
        text_extractor=text_ext,
        risk_filter=risk_filter,
        caption_encoder=cap_encoder,
        caption_decoder=cap_decoder,
        univid_lite=lite,
        univid_rag=rag,
        case_library=case_lib,
        trend_system=trend_sys,
        device=device,
    )
