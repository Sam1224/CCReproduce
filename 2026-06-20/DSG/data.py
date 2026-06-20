"""Toy offline corpus + queries for DSG.

Mirrors the paper's evaluation surfaces at toy scale:
  * a public-QA-style factoid set (SimpleQA / FreshQA / HotpotQA flavour)
  * an e-commerce Query-Intent-Understanding (QIU) set with intent labels

Every document carries (url, domain, snippet, answer) so the gateway can do
source-aware rendering and per-domain TTL, and the eval can score accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class Document:
    doc_id: str
    url: str
    domain: str          # used for per-domain TTL
    snippet: str
    answer: str
    keywords: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class Query:
    qid: str
    text: str
    gold_answer: str
    intent: str          # QIU-style query-intent label
    doc_id: str          # the document that grounds the answer
    recency: str = "static"   # "static" | "fresh" -> drives TTL policy


# --- tiny grounding corpus ----------------------------------------------------
CORPUS: List[Document] = [
    Document("d_capital_fr", "https://en.wikipedia.org/wiki/Paris", "wikipedia.org",
             "Paris is the capital and most populous city of France.", "Paris",
             ["capital", "france"]),
    Document("d_capital_jp", "https://en.wikipedia.org/wiki/Tokyo", "wikipedia.org",
             "Tokyo is the capital city of Japan and its largest metropolis.", "Tokyo",
             ["capital", "japan"]),
    Document("d_tallest", "https://en.wikipedia.org/wiki/Mount_Everest", "wikipedia.org",
             "Mount Everest is Earth's highest mountain above sea level.", "Mount Everest",
             ["tallest", "mountain"]),
    Document("d_relativity", "https://en.wikipedia.org/wiki/Theory_of_relativity", "wikipedia.org",
             "The theory of relativity was developed by Albert Einstein.", "Albert Einstein",
             ["relativity", "physics"]),
    Document("d_python", "https://docs.python.org/3/", "python.org",
             "Python is a high-level programming language created by Guido van Rossum.",
             "Guido van Rossum", ["python", "language"]),
    Document("d_ddash_founded", "https://about.doordash.com/", "doordash.com",
             "DoorDash, a food delivery company, was founded in 2013.", "2013",
             ["doordash", "founded"]),
    # e-commerce / QIU style catalog evidence
    Document("d_qiu_airpods", "https://store.example.com/airpods", "store.example.com",
             "AirPods Pro are wireless noise-cancelling earbuds, an electronics product.",
             "Electronics", ["airpods", "earbuds"]),
    Document("d_qiu_advil", "https://store.example.com/advil", "store.example.com",
             "Advil is an over-the-counter ibuprofen pain reliever (health & pharmacy).",
             "Health", ["advil", "pain", "medicine"]),
    Document("d_qiu_diapers", "https://store.example.com/pampers", "store.example.com",
             "Pampers diapers are a baby-care retail product.", "Baby",
             ["diapers", "baby"]),
    Document("d_qiu_milk", "https://store.example.com/milk", "store.example.com",
             "Organic whole milk is a grocery dairy product.", "Grocery",
             ["milk", "grocery"]),
    # fresh / recency-sensitive (short TTL)
    Document("d_weather_sf", "https://weather.example.com/sf", "weather.example.com",
             "Current weather in San Francisco: 18C and foggy.", "18C",
             ["weather", "san francisco"]),
    Document("d_stock_acme", "https://finance.example.com/acme", "finance.example.com",
             "ACME Corp stock is trading at $42 today.", "$42",
             ["stock", "acme", "price"]),
]

DOC_BY_ID = {d.doc_id: d for d in CORPUS}


# --- queries: base + paraphrases (paraphrases drive semantic-cache reuse) ------
QUERIES: List[Query] = [
    Query("q1", "What is the capital of France?", "Paris", "factoid", "d_capital_fr"),
    Query("q2", "What is the capital of Japan?", "Tokyo", "factoid", "d_capital_jp"),
    Query("q3", "What is the tallest mountain on Earth?", "Mount Everest", "factoid", "d_tallest"),
    Query("q4", "Who developed the theory of relativity?", "Albert Einstein", "factoid", "d_relativity"),
    Query("q5", "Who created the Python programming language?", "Guido van Rossum", "factoid", "d_python"),
    Query("q6", "In what year was DoorDash founded?", "2013", "factoid", "d_ddash_founded"),
    Query("q7", "What product category are AirPods Pro?", "Electronics", "qiu_intent", "d_qiu_airpods"),
    Query("q8", "What category does Advil belong to?", "Health", "qiu_intent", "d_qiu_advil"),
    Query("q9", "Which category are Pampers diapers?", "Baby", "qiu_intent", "d_qiu_diapers"),
    Query("q10", "What category is organic whole milk?", "Grocery", "qiu_intent", "d_qiu_milk"),
    Query("q11", "What is the current weather in San Francisco?", "18C", "factoid", "d_weather_sf", "fresh"),
    Query("q12", "What is ACME Corp stock price today?", "$42", "qiu_intent", "d_stock_acme", "fresh"),
]


# Paraphrases of base queries -> SAME doc/answer. The semantic cache should reuse
# the base result for these (positive pairs for tau calibration).
PARAPHRASES: List[Query] = [
    Query("p1", "The capital city of France is what?", "Paris", "factoid", "d_capital_fr"),
    Query("p2", "What is the capital city of Japan?", "Tokyo", "factoid", "d_capital_jp"),
    Query("p3", "What is the highest mountain on Earth?", "Mount Everest", "factoid", "d_tallest"),
    Query("p4", "Who developed the relativity theory?", "Albert Einstein", "factoid", "d_relativity"),
    Query("p5", "Who created the Python programming language originally?", "Guido van Rossum", "factoid", "d_python"),
    Query("p7", "What product category are the AirPods Pro?", "Electronics", "qiu_intent", "d_qiu_airpods"),
    Query("p9", "Which product category are Pampers diapers?", "Baby", "qiu_intent", "d_qiu_diapers"),
    Query("p10", "What product category is organic whole milk?", "Grocery", "qiu_intent", "d_qiu_milk"),
]


def all_queries() -> List[Query]:
    return list(QUERIES)


def calibration_pairs():
    """Return (positives, negatives) for semantic-cache tau calibration.

    positives: (paraphrase, base) that SHOULD reuse the cache (same answer).
    negatives: (other_base, base) that should NOT reuse (different answer).
    """
    base_by_doc = {q.doc_id: q for q in QUERIES}
    positives = [(p, base_by_doc[p.doc_id]) for p in PARAPHRASES if p.doc_id in base_by_doc]
    negatives = []
    for i, a in enumerate(QUERIES):
        b = QUERIES[(i + 1) % len(QUERIES)]   # a different query
        if a.doc_id != b.doc_id:
            negatives.append((a, b))
    return positives, negatives
