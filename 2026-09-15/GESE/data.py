from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset

# -----------------------------
# Vocabulary (toy headlines)
# headline = [BOS, STYLE, TOPIC, FILLER, EOS]
# -----------------------------

PAD = 0
BOS = 1
EOS = 2

TOPICS = ["sports", "finance", "tech", "travel", "food"]
STYLES = ["neutral", "exciting", "serious", "funny", "mystery"]
FILLERS = ["today", "guide", "tips", "story", "insights"]

TOPIC_OFFSET = 3
STYLE_OFFSET = TOPIC_OFFSET + len(TOPICS)
FILLER_OFFSET = STYLE_OFFSET + len(STYLES)
VOCAB_SIZE = FILLER_OFFSET + len(FILLERS)

TOPIC_TOKEN_IDS = list(range(TOPIC_OFFSET, TOPIC_OFFSET + len(TOPICS)))
STYLE_TOKEN_IDS = list(range(STYLE_OFFSET, STYLE_OFFSET + len(STYLES)))
FILLER_TOKEN_IDS = list(range(FILLER_OFFSET, FILLER_OFFSET + len(FILLERS)))


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _build_id_map(size: int, pairs: List[Tuple[int, int]]) -> torch.Tensor:
    """Build a dense lookup table: vocab_id -> index in a category, else -1."""

    table = torch.full((size,), -1, dtype=torch.long)
    for token_id, idx in pairs:
        table[token_id] = idx
    return table


VOCAB_TO_TOPIC = _build_id_map(VOCAB_SIZE, [(tid, i) for i, tid in enumerate(TOPIC_TOKEN_IDS)])
VOCAB_TO_STYLE = _build_id_map(VOCAB_SIZE, [(sid, i) for i, sid in enumerate(STYLE_TOKEN_IDS)])


@dataclass
class Splits:
    train_pairs: List[Tuple[int, int]]
    test_pairs: List[Tuple[int, int]]


def build_splits(
    seed: int,
    n_users: int,
    n_items: int,
    n_train: int,
    n_test: int,
) -> Splits:
    rng = random.Random(seed)
    train_pairs = [(rng.randrange(n_users), rng.randrange(n_items)) for _ in range(n_train)]
    test_pairs = [(rng.randrange(n_users), rng.randrange(n_items)) for _ in range(n_test)]
    return Splits(train_pairs=train_pairs, test_pairs=test_pairs)


@dataclass
class SyntheticWorld:
    """A synthetic click environment.

    - Each item has exactly one topic.
    - User has personalized preference over topic faithfulness and style.
    - Reward is a sigmoid CTR proxy in [0, 1].
    """

    user_topic_pref: torch.Tensor  # [U, T] in [0,1]
    user_style_pref: torch.Tensor  # [U, S] in [0,1]
    item_topic: torch.Tensor  # [I] in [0..T-1]

    @property
    def n_users(self) -> int:
        return int(self.user_topic_pref.shape[0])

    @property
    def n_items(self) -> int:
        return int(self.item_topic.shape[0])

    @property
    def n_topics(self) -> int:
        return int(self.user_topic_pref.shape[1])

    @property
    def n_styles(self) -> int:
        return int(self.user_style_pref.shape[1])


def build_world(n_users: int = 64, n_items: int = 200, seed: int = 7) -> SyntheticWorld:
    rng = torch.Generator().manual_seed(seed)

    # Topic preference: each user likes 1~2 topics strongly, others weak.
    user_topic_pref = torch.rand((n_users, len(TOPICS)), generator=rng) * 0.2
    top_topic = torch.randint(0, len(TOPICS), (n_users,), generator=rng)
    user_topic_pref[torch.arange(n_users), top_topic] += 0.7

    # Style preference: each user has one favorite style.
    user_style_pref = torch.rand((n_users, len(STYLES)), generator=rng) * 0.2
    fav_style = torch.randint(0, len(STYLES), (n_users,), generator=rng)
    user_style_pref[torch.arange(n_users), fav_style] += 0.8

    item_topic = torch.randint(0, len(TOPICS), (n_items,), generator=rng)
    return SyntheticWorld(
        user_topic_pref=user_topic_pref.clamp(0.0, 1.0),
        user_style_pref=user_style_pref.clamp(0.0, 1.0),
        item_topic=item_topic,
    )


def static_headline(item_topic_id: int) -> torch.Tensor:
    """Non-personalized baseline headline."""

    neutral_style = STYLE_OFFSET + STYLES.index("neutral")
    today_filler = FILLER_OFFSET + FILLERS.index("today")
    topic_token = TOPIC_OFFSET + int(item_topic_id)
    return torch.tensor([BOS, neutral_style, topic_token, today_filler, EOS], dtype=torch.long)


def reward_batch(world: SyntheticWorld, user_ids: torch.Tensor, item_ids: torch.Tensor, titles: torch.Tensor) -> torch.Tensor:
    """Vectorized CTR proxy.

    Args:
        user_ids: [N]
        item_ids: [N]
        titles: [N, L], L==5

    Returns:
        ctr: [N] in [0,1]
    """

    style_token = titles[:, 1].clamp(0, VOCAB_SIZE - 1)
    topic_token = titles[:, 2].clamp(0, VOCAB_SIZE - 1)

    vocab_to_style = VOCAB_TO_STYLE.to(style_token.device)
    vocab_to_topic = VOCAB_TO_TOPIC.to(topic_token.device)
    style_idx = vocab_to_style[style_token]
    topic_idx = vocab_to_topic[topic_token]

    item_topic = world.item_topic[item_ids]

    style_valid = (style_idx >= 0).float()
    topic_valid = (topic_idx >= 0).float()
    topic_match = (topic_idx == item_topic).float() * topic_valid

    safe_style_idx = style_idx.clamp(min=0)
    style_score = world.user_style_pref[user_ids, safe_style_idx] * style_valid

    # If the title expresses the correct topic, user gains its preference on that topic.
    topic_score = world.user_topic_pref[user_ids, item_topic] * topic_match

    # CTR logit (faithfulness > style)
    logit = 2.4 * topic_score + 1.3 * style_score - 1.3
    ctr = torch.sigmoid(logit)
    return ctr


def diversity_reward(titles_bkl: torch.Tensor) -> torch.Tensor:
    """Set-level diversity reward.

    titles_bkl: [B, K, L]
    returns: [B] in [0,1]

    We compute average pairwise Jaccard distance on token sets (excluding specials).
    """

    bsz, k, _ = titles_bkl.shape
    outs = []
    for b in range(bsz):
        sets = []
        for j in range(k):
            toks = [int(x) for x in titles_bkl[b, j].tolist() if x not in (PAD, BOS, EOS)]
            sets.append(set(toks))
        if k <= 1:
            outs.append(0.0)
            continue
        dist_sum = 0.0
        cnt = 0
        for i in range(k):
            for j in range(i + 1, k):
                a, c = sets[i], sets[j]
                inter = len(a & c)
                union = max(1, len(a | c))
                dist_sum += 1.0 - inter / union
                cnt += 1
        outs.append(dist_sum / max(1, cnt))
    return torch.tensor(outs, dtype=torch.float32)


class PairDataset(Dataset):
    """Dataset of (user_id, item_id)."""

    def __init__(self, pairs: List[Tuple[int, int]]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        u, it = self.pairs[idx]
        return {"user_id": torch.tensor(u, dtype=torch.long), "item_id": torch.tensor(it, dtype=torch.long)}


class SelectorDataset(Dataset):
    """Flattened candidate dataset for selector offline training."""

    def __init__(self, user_ids: torch.Tensor, item_ids: torch.Tensor, titles: torch.Tensor, rewards: torch.Tensor):
        assert user_ids.ndim == 1
        assert item_ids.ndim == 1
        assert titles.ndim == 2
        assert rewards.ndim == 1
        assert len(user_ids) == len(item_ids) == len(titles) == len(rewards)
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.titles = titles
        self.rewards = rewards

    def __len__(self) -> int:
        return int(self.user_ids.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "user_id": self.user_ids[idx],
            "item_id": self.item_ids[idx],
            "title": self.titles[idx],
            "reward": self.rewards[idx],
        }
