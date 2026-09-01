from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str


@dataclass(frozen=True)
class QueryExample:
    query: str
    answer_relation: str
    expected_doc_id: str
    expected_object: str


def build_toy_corpus() -> Tuple[List[Document], List[QueryExample]]:
    documents = [
        Document(
            doc_id="product_claims",
            text=(
                "Creator Alpha promotes GlowMask and claims it cures acne in seven days. "
                "The review team flags the cure claim because cosmetics may improve appearance but cannot treat disease. "
                "A verified product sheet says GlowMask hydrates skin and reduces visible oiliness."
            ),
        ),
        Document(
            doc_id="duplicate_video",
            text=(
                "Creator Beta uploads a short video with the same opening frame as a banned clip. "
                "The multimodal encoder matches the caption and thumbnail to a previous violation. "
                "Governance policy requires clustering similar uploads before reviewer escalation."
            ),
        ),
        Document(
            doc_id="risky_coupon",
            text=(
                "Shop Gamma advertises a coupon stack that states every user receives a free phone. "
                "Order logs show only lottery winners receive the phone while all users get a small discount. "
                "The misleading benefit description should be rewritten before campaign launch."
            ),
        ),
        Document(
            doc_id="attribute_cleanup",
            text=(
                "The catalog pipeline extracts sleeve length, material, color and size from item images and titles. "
                "Noisy captions often mix model height with product size, which hurts vector retrieval. "
                "Attribute normalization maps aliases such as tee and t shirt into one category."
            ),
        ),
    ]
    queries = [
        QueryExample(
            query="Which creator made an unsupported medical claim about GlowMask?",
            answer_relation="flags",
            expected_doc_id="product_claims",
            expected_object="cure claim",
        ),
        QueryExample(
            query="How should near-duplicate banned creator videos be handled?",
            answer_relation="requires",
            expected_doc_id="duplicate_video",
            expected_object="clustering similar uploads",
        ),
        QueryExample(
            query="What is misleading in the coupon promotion?",
            answer_relation="receives",
            expected_doc_id="risky_coupon",
            expected_object="free phone",
        ),
        QueryExample(
            query="What improves product attribute retrieval quality?",
            answer_relation="maps",
            expected_doc_id="attribute_cleanup",
            expected_object="aliases",
        ),
    ]
    return documents, queries


def relation_lexicon() -> Dict[str, str]:
    return {
        "claims": "claim",
        "flags": "moderation",
        "requires": "policy",
        "matches": "similarity",
        "receives": "promotion",
        "rewritten": "correction",
        "extracts": "attribute",
        "maps": "normalization",
        "hurts": "quality",
    }


def domain_entities() -> Sequence[str]:
    return [
        "Creator Alpha",
        "GlowMask",
        "Creator Beta",
        "Shop Gamma",
        "coupon",
        "caption",
        "thumbnail",
        "attribute",
        "sleeve length",
        "material",
        "color",
        "size",
        "vector retrieval",
        "governance policy",
    ]
