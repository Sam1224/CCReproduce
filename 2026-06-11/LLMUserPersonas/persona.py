from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9:\s]+", " ", text)
    return [t for t in text.split() if t]


@dataclass(frozen=True)
class Persona:
    kind: str  # "summary" | "explore"
    text: str
    evidence_video_ids: list[int]
    confidence: float


class RuleBasedPersonaGenerator:
    def __init__(self, topic_vec: dict[str, np.ndarray]):
        self.topic_vec = topic_vec

    def generate(self, videos: list[dict], cluster_video_ids: list[int], history_video_ids: set[int]) -> list[Persona]:
        cluster = [videos[i] for i in cluster_video_ids]

        topics = [v["topic"] for v in cluster]
        topic = max(set(topics), key=topics.count)

        # summarized interest
        subtopics = [v["subtopic"] for v in cluster]
        sub = max(set(subtopics), key=subtopics.count)
        summary = Persona(
            kind="summary",
            text=f"{topic} · {sub}",
            evidence_video_ids=cluster_video_ids[:3],
            confidence=min(0.95, 0.5 + topics.count(topic) / max(1, len(topics))),
        )

        # exploration interest: pick a nearby topic not already heavily consumed
        history_topics = [videos[i]["topic"] for i in history_video_ids]
        hist_counts = {t: history_topics.count(t) for t in set(history_topics)}

        base = self.topic_vec[topic]
        cand = []
        for t, vec in self.topic_vec.items():
            if t == topic:
                continue
            score = float(base @ vec)
            penalty = 0.15 * hist_counts.get(t, 0)
            cand.append((score - penalty, t))

        cand.sort(reverse=True)
        explore_topic = cand[0][1] if cand else topic
        explore = Persona(
            kind="explore",
            text=f"Try: {explore_topic}",
            evidence_video_ids=cluster_video_ids[:2],
            confidence=0.55,
        )

        return [summary, explore]


class LLMPersonaGenerator:
    """Placeholder interface for the paper's LLM-based persona generation.

    In the full system, this module would:

    1) Build a structured prompt from a cluster of watched videos (titles, metadata, embeddings, etc.)
    2) Call a teacher LLM to generate:
       - summarized interest (what the user already likes)
       - exploration interest (novel but relevant topics)
       plus reasoning traces
    3) Distill the teacher into a small student model for online serving
    4) Apply safety filters / formatting checks

    This repo keeps the interface so the caller can swap in an internal LLM client.
    """

    def generate(self, videos: list[dict], cluster_video_ids: list[int], history_video_ids: set[int]) -> list[Persona]:
        raise NotImplementedError("Replace with internal/open-source LLM client")


def retrieve_candidates_by_persona(
    persona_text: str,
    videos: list[dict],
    topic_vec: dict[str, np.ndarray],
    top_k: int = 20,
) -> list[int]:
    tokens = _tokenize(persona_text)
    # crude mapping: if a topic token exists, use its embedding
    vecs = []
    for t in topic_vec:
        if t in tokens:
            vecs.append(topic_vec[t])

    if not vecs:
        # fallback to a random direction
        q = np.zeros_like(next(iter(topic_vec.values())))
        q[0] = 1.0
    else:
        q = np.stack(vecs, axis=0).mean(axis=0)
        q = q / (np.linalg.norm(q) + 1e-12)

    mat = np.stack([v["emb"] for v in videos], axis=0)
    scores = mat @ q
    idx = np.argsort(-scores)[:top_k]
    return [int(i) for i in idx.tolist()]
