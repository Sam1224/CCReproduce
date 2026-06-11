from dataclasses import dataclass

import numpy as np


def set_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


@dataclass(frozen=True)
class ToyPersonaConfig:
    num_videos: int = 2000
    embed_dim: int = 64
    num_users: int = 200
    history_len: int = 80


TOPICS = {
    "fashion": ["streetwear", "outfit", "sneakers", "summer"],
    "makeup": ["skincare", "lipstick", "foundation", "review"],
    "cooking": ["airfryer", "noodles", "dessert", "mealprep"],
    "gaming": ["fps", "moba", "speedrun", "review"],
    "pets": ["cats", "dogs", "training", "funny"],
    "travel": ["budget", "hotel", "food", "itinerary"],
    "fitness": ["hiit", "yoga", "running", "nutrition"],
    "tech": ["ai", "phone", "laptop", "tutorial"],
    "finance": ["stocks", "budgeting", "crypto", "tax"],
    "parenting": ["toddlers", "school", "tips", "toys"],
    "diy": ["home", "craft", "tools", "decor"],
    "music": ["covers", "guitar", "pop", "practice"],
}


def build_catalog(cfg: ToyPersonaConfig, seed: int = 7):
    rng = set_seed(seed)

    topic_names = list(TOPICS.keys())
    topic_vec = rng.normal(size=(len(topic_names), cfg.embed_dim)).astype(np.float32)
    topic_vec = topic_vec / (np.linalg.norm(topic_vec, axis=1, keepdims=True) + 1e-12)

    videos = []
    for vid in range(cfg.num_videos):
        t_idx = int(rng.integers(0, len(topic_names)))
        topic = topic_names[t_idx]
        sub = TOPICS[topic][int(rng.integers(0, len(TOPICS[topic])))]

        emb = topic_vec[t_idx] + rng.normal(scale=0.15, size=(cfg.embed_dim,)).astype(np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-12)

        title = f"{topic}: {sub}"
        videos.append({"id": vid, "topic": topic, "subtopic": sub, "title": title, "emb": emb})

    return {"videos": videos, "topic_vec": {t: topic_vec[i] for i, t in enumerate(topic_names)}}


def build_users(cfg: ToyPersonaConfig, catalog: dict, seed: int = 7):
    rng = set_seed(seed + 1)

    videos = catalog["videos"]
    topic_names = list(TOPICS.keys())

    users = []
    for u in range(cfg.num_users):
        # each user has 2-3 main topics
        main_topics = rng.choice(topic_names, size=int(rng.integers(2, 4)), replace=False)
        weights = rng.random(size=len(main_topics))
        weights = weights / weights.sum()

        # sample history
        hist = []
        for _ in range(cfg.history_len):
            t = str(rng.choice(main_topics, p=weights))
            candidates = [v for v in videos if v["topic"] == t]
            hist.append(int(rng.choice([v["id"] for v in candidates])))

        users.append({"user_id": u, "main_topics": main_topics.tolist(), "history": hist})

    return users
