from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class ToyDoc:
    doc_id: str
    text: str


@dataclass(frozen=True)
class ToyQuery:
    query_id: str
    query: str
    answer: str
    gold_doc_id: str
    gold_span: str


def build_toy_corpus() -> Tuple[List[ToyDoc], List[ToyQuery]]:
    """Create a small corpus with intentionally redundant content.

    The goal is not to mimic any real benchmark, but to provide:
    - Documents that become redundant after overlap chunking
    - Queries with a well-defined gold supporting span
    """

    docs = [
        ToyDoc(
            doc_id="doc_0",
            text=(
                "BrandX Running Shoes are lightweight and breathable. "
                "BrandX Running Shoes are lightweight and breathable. "
                "They use FoamPlus cushioning for long-distance comfort. "
                "Return policy: 30 days with receipt. "
                "Materials: recycled mesh upper and rubber outsole."
            ),
        ),
        ToyDoc(
            doc_id="doc_1",
            text=(
                "Creator Alice posted a review video about BrandY Camera. "
                "Creator Alice posted a review video about BrandY Camera. "
                "She claims 4K60 recording and strong stabilization. "
                "Policy: Sponsored content must be disclosed clearly. "
                "Violation examples include hidden ads and misleading claims."
            ),
        ),
        ToyDoc(
            doc_id="doc_2",
            text=(
                "ProductZ Kitchen Knife is made of stainless steel. "
                "It has a 20cm blade and ergonomic handle. "
                "Safety note: do not promote self-harm or violence. "
                "Content moderation requires context and intent. "
                "The same word can be acceptable in one community and harmful in another."
            ),
        ),
    ]

    queries = [
        ToyQuery(
            query_id="q0",
            query="What is the return policy for BrandX Running Shoes?",
            answer="30 days with receipt",
            gold_doc_id="doc_0",
            gold_span="Return policy: 30 days with receipt",
        ),
        ToyQuery(
            query_id="q1",
            query="What must be disclosed for sponsored content in Alice's video?",
            answer="Sponsored content must be disclosed clearly",
            gold_doc_id="doc_1",
            gold_span="Policy: Sponsored content must be disclosed clearly",
        ),
        ToyQuery(
            query_id="q2",
            query="Why does content moderation require context and intent?",
            answer="Because acceptability depends on community and usage",
            gold_doc_id="doc_2",
            gold_span="Content moderation requires context and intent",
        ),
    ]

    return docs, queries


def as_doc_tuples(docs: List[ToyDoc]) -> List[Tuple[str, str]]:
    return [(d.doc_id, d.text) for d in docs]


def build_gold_reference(queries: List[ToyQuery]) -> Dict[str, str]:
    """Map query_id -> gold span text for token-level retrieval evaluation."""

    return {q.query_id: q.gold_span for q in queries}
