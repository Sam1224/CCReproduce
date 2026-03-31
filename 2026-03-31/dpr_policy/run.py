from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class Doc:
    doc_id: str
    text: str


TOPICS = {
    "hate_speech": ["hate", "slur", "racist", "protected", "group"],
    "sexual_content": ["sex", "porn", "explicit"],
    "violence": ["kill", "weapon", "violence"],
    "harassment": ["bully", "harass", "threat"],
    "fraud": ["scam", "fraud", "phishing"],
}


def toy_corpus() -> List[Doc]:
    return [
        Doc("d1", "We prohibit hate speech targeting protected groups and any racist slurs."),
        Doc("d2", "Sexually explicit content and pornography are disallowed in ads and queries."),
        Doc("d3", "Violence, weapons, and instructions to harm others are strictly prohibited."),
        Doc("d4", "Harassment and threats, including bullying individuals, are not allowed."),
        Doc("d5", "Fraud, scams, and phishing attempts are forbidden and should be blocked."),
        Doc("d6", "Context matters: some words may be allowed in educational or news contexts."),
    ]


def tokenize(text: str) -> List[str]:
    out = []
    for t in text.lower().replace(".", " ").replace(",", " ").split():
        if t:
            out.append(t)
    return out


class SimpleTfidfRetriever:
    def __init__(self, docs: List[Doc]) -> None:
        self.docs = docs
        self.doc_tokens = [tokenize(d.text) for d in docs]
        self.vocab = sorted({t for toks in self.doc_tokens for t in toks})
        self.t2i = {t: i for i, t in enumerate(self.vocab)}

        df = np.zeros(len(self.vocab), dtype=np.float64)
        for toks in self.doc_tokens:
            for t in set(toks):
                df[self.t2i[t]] += 1
        self.idf = np.log((1 + len(docs)) / (1 + df)) + 1.0

        self.doc_vecs = np.stack([self._tfidf_vec(toks) for toks in self.doc_tokens], axis=0)

    def _tfidf_vec(self, toks: List[str]) -> np.ndarray:
        tf = np.zeros(len(self.vocab), dtype=np.float64)
        for t in toks:
            if t in self.t2i:
                tf[self.t2i[t]] += 1
        if tf.sum() > 0:
            tf = tf / tf.sum()
        v = tf * self.idf
        n = np.linalg.norm(v)
        return v / (n + 1e-12)

    def topk(self, query: str, k: int = 3) -> List[Tuple[Doc, float]]:
        qv = self._tfidf_vec(tokenize(query))
        sims = self.doc_vecs @ qv
        idx = np.argsort(-sims)[:k]
        return [(self.docs[int(i)], float(sims[int(i)])) for i in idx]


def synthesize_policy_section(topic: str, evidence: List[Tuple[Doc, float]]) -> str:
    bullets = []
    for d, s in evidence:
        bullets.append(f"- ({d.doc_id}, sim={s:.3f}) {d.text}")

    kw = ", ".join(TOPICS.get(topic, [])[:5])
    return (
        f"### {topic}\n\n"
        f"**Keywords:** {kw}\n\n"
        f"**Rules (from sources):**\n" + "\n".join(bullets) + "\n\n"
        f"**Enforcement:** block content that matches the above patterns unless a documented exception applies.\n"
    )


def build_policy(docs: List[Doc]) -> str:
    r = SimpleTfidfRetriever(docs)
    out = ["# Safety Policy (constructed)", ""]
    for topic, kws in TOPICS.items():
        query = " ".join(kws)
        evidence = r.topk(query, k=3)
        out.append(synthesize_policy_section(topic, evidence))
    return "\n".join(out)


def coverage_eval(policy: str, n: int = 200) -> Dict[str, float]:
    hits = 0
    for _ in range(n):
        topic = random.choice(list(TOPICS.keys()))
        ok = any(kw in policy.lower() for kw in TOPICS[topic])
        hits += int(ok)
    return {"coverage": hits / max(1, n)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="policy.md")
    args = ap.parse_args()

    docs = toy_corpus()
    policy = build_policy(docs)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(policy)

    metrics = coverage_eval(policy)
    print({"saved": args.out, **metrics})


if __name__ == "__main__":
    main()
