from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from kb import KnowledgeBase, tokenize


@dataclass
class Retrieved:
    doc_id: str
    score: float
    why: str


def bow_score(query_tokens: List[str], doc_tokens: List[str]) -> float:
    if not query_tokens:
        return 0.0
    q = set(query_tokens)
    d = set(doc_tokens)
    return len(q & d) / (len(q) ** 0.5 * max(1.0, len(d) ** 0.5))


def extract_entities(query: str) -> List[str]:
    # toy heuristic: treat CamelCase tokens as entities
    out: List[str] = []
    for tok in query.split():
        t = tok.strip(".,!?()[]{}\"'")
        if len(t) >= 2 and any(ch.isupper() for ch in t[1:]):
            out.append(t)
    return out


def low_level_retrieve(kb: KnowledgeBase, query: str, *, k: int = 3) -> List[Retrieved]:
    qt = tokenize(query)
    scored: List[Retrieved] = []
    for d in kb.docs.values():
        s = bow_score(qt, tokenize(d.text))
        if s > 0:
            scored.append(Retrieved(doc_id=d.doc_id, score=s, why="bow"))
    scored.sort(key=lambda x: x.score, reverse=True)
    return scored[:k]


def high_level_retrieve(kb: KnowledgeBase, query: str, *, hops: int = 2, k: int = 4) -> List[Retrieved]:
    ents = extract_entities(query)
    if not ents:
        return []

    frontier: Set[str] = set(ents)
    visited: Set[str] = set(ents)
    for _ in range(hops):
        nxt: Set[str] = set()
        for e in frontier:
            for nb in kb.neighbors(e):
                if nb not in visited:
                    visited.add(nb)
                    nxt.add(nb)
        frontier = nxt

    # score docs by how many visited entities they contain
    scored: List[Retrieved] = []
    for d in kb.docs.values():
        hit = len(set(d.entities) & visited)
        if hit > 0:
            scored.append(Retrieved(doc_id=d.doc_id, score=float(hit), why="graph"))
    scored.sort(key=lambda x: x.score, reverse=True)
    return scored[:k]


def fuse(low: List[Retrieved], high: List[Retrieved], *, k: int = 5) -> List[Retrieved]:
    agg: Dict[str, Retrieved] = {}
    for r in low + high:
        prev = agg.get(r.doc_id)
        if not prev:
            agg[r.doc_id] = r
            continue
        agg[r.doc_id] = Retrieved(doc_id=r.doc_id, score=prev.score + r.score, why=prev.why + "+" + r.why)

    out = list(agg.values())
    out.sort(key=lambda x: x.score, reverse=True)
    return out[:k]


def retrieve(kb: KnowledgeBase, query: str) -> List[Retrieved]:
    low = low_level_retrieve(kb, query, k=3)
    high = high_level_retrieve(kb, query, hops=2, k=4)
    return fuse(low, high, k=5)
