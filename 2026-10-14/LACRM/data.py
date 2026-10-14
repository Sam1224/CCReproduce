from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


VIEW_NAMES = [
    "direct_homogeneous",
    "intra_brand_structural",
    "intra_brand_cross_channel",
    "cross_category_substitution",
    "temporal_campaign",
]


@dataclass
class MarketConfig:
    n_advertisers: int = 180
    n_categories: int = 8
    n_brands: int = 24
    n_intents: int = 10
    n_styles: int = 6
    n_audiences: int = 6
    n_prices: int = 5
    n_channels: int = 4
    behavior_dim: int = 12
    seed: int = 7


@dataclass
class ToyMarket:
    category: np.ndarray
    brand: np.ndarray
    intent: np.ndarray
    style: np.ndarray
    audience: np.ndarray
    price: np.ndarray
    channel: np.ndarray
    merchant_tier: np.ndarray
    festival_active: np.ndarray
    behavior: np.ndarray
    train_view_scores: np.ndarray  # [N,N,5]
    eval_view_scores: np.ndarray   # [N,N,5]
    train_scores: np.ndarray       # [N,N]
    eval_scores: np.ndarray        # [N,N]


def set_seed(seed: int) -> None:
    np.random.seed(seed)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def make_behavior_features(
    strength: np.ndarray,
    category: np.ndarray,
    price: np.ndarray,
    festival_active: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    n = strength.shape[0]
    cat_center = category / max(1, int(category.max()))
    price_center = price / max(1, int(price.max()))

    # six groups × two dims = 12 dims
    traffic = np.stack(
        [
            np.log1p(20 + 80 * strength + 10 * rng.normal(size=n)),
            np.log1p(15 + 70 * strength + 8 * rng.normal(size=n)),
        ],
        axis=1,
    )
    conversion = np.stack(
        [
            sigmoid(1.2 * strength - 0.2 * price_center + 0.3 * rng.normal(size=n)),
            sigmoid(1.0 * strength - 0.3 * price_center + 0.3 * rng.normal(size=n)),
        ],
        axis=1,
    )
    dynamic = np.stack(
        [
            0.7 * strength + 0.2 * cat_center + 0.1 * rng.normal(size=n),
            0.6 * strength + 0.3 * price_center + 0.1 * rng.normal(size=n),
        ],
        axis=1,
    )
    lifecycle = np.stack(
        [
            0.5 * strength + 0.2 * festival_active + 0.15 * rng.normal(size=n),
            0.4 * strength + 0.25 * cat_center + 0.15 * rng.normal(size=n),
        ],
        axis=1,
    )
    quality = np.stack(
        [
            sigmoid(1.0 * strength + 0.1 * rng.normal(size=n)),
            sigmoid(0.8 * strength + 0.2 * festival_active + 0.1 * rng.normal(size=n)),
        ],
        axis=1,
    )
    category_trend = np.stack(
        [
            0.35 * cat_center + 0.35 * strength + 0.15 * rng.normal(size=n),
            0.25 * price_center + 0.45 * strength + 0.15 * rng.normal(size=n),
        ],
        axis=1,
    )

    behavior = np.concatenate(
        [traffic, conversion, dynamic, lifecycle, quality, category_trend], axis=1
    ).astype(np.float32)
    behavior = (behavior - behavior.mean(axis=0, keepdims=True)) / (
        behavior.std(axis=0, keepdims=True) + 1e-6
    )
    return behavior.astype(np.float32)


def _topk_mask(scores: np.ndarray, k: int, threshold: float) -> np.ndarray:
    n = scores.shape[0]
    mask = np.zeros_like(scores, dtype=np.float32)
    for i in range(n):
        order = np.argsort(scores[i])[::-1]
        keep = [j for j in order if j != i and scores[i, j] > threshold][:k]
        mask[i, keep] = 1.0
    return mask


def generate_market(cfg: MarketConfig) -> ToyMarket:
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_advertisers

    category = rng.integers(0, cfg.n_categories, size=n, dtype=np.int64)
    brand = rng.integers(0, cfg.n_brands, size=n, dtype=np.int64)
    intent = rng.integers(0, cfg.n_intents, size=n, dtype=np.int64)
    style = rng.integers(0, cfg.n_styles, size=n, dtype=np.int64)
    audience = rng.integers(0, cfg.n_audiences, size=n, dtype=np.int64)
    price = rng.integers(0, cfg.n_prices, size=n, dtype=np.int64)
    channel = rng.integers(0, cfg.n_channels, size=n, dtype=np.int64)
    merchant_tier = rng.integers(0, 4, size=n, dtype=np.int64)
    festival_active = rng.integers(0, 2, size=n, dtype=np.int64)

    strength = (
        0.45 * merchant_tier
        + 0.20 * (cfg.n_prices - 1 - price)
        + 0.15 * (brand % 5)
        + 0.20 * rng.normal(size=n)
    ).astype(np.float32)
    strength = (strength - strength.min()) / (strength.max() - strength.min() + 1e-6)

    behavior = make_behavior_features(strength, category, price, festival_active, rng)

    same_category = (category[:, None] == category[None, :]).astype(np.float32)
    same_brand = (brand[:, None] == brand[None, :]).astype(np.float32)
    same_intent = (intent[:, None] == intent[None, :]).astype(np.float32)
    same_style = (style[:, None] == style[None, :]).astype(np.float32)
    same_audience = (audience[:, None] == audience[None, :]).astype(np.float32)
    diff_category = 1.0 - same_category
    diff_channel = (channel[:, None] != channel[None, :]).astype(np.float32)
    price_close = 1.0 - np.abs(price[:, None] - price[None, :]) / max(1, cfg.n_prices - 1)
    both_festival = (festival_active[:, None] * festival_active[None, :]).astype(np.float32)

    directional = sigmoid(2.0 * (strength[None, :] - strength[:, None])).astype(np.float32)

    view_direct = (
        0.40 * same_category
        + 0.25 * same_intent
        + 0.20 * price_close
        + 0.15 * same_style
    ) * (0.55 + 0.45 * directional)

    view_structural = (
        same_brand
        * diff_category
        * (0.45 + 0.30 * same_audience + 0.25 * price_close)
    ) * (0.50 + 0.50 * directional)

    view_cross_channel = (
        same_brand
        * same_category
        * diff_channel
        * (0.55 + 0.25 * same_intent + 0.20 * price_close)
    ) * (0.40 + 0.60 * directional)

    view_substitution = (
        diff_category
        * (
            0.35 * same_audience
            + 0.25 * same_intent
            + 0.20 * price_close
            + 0.20 * same_style
        )
    ) * (0.45 + 0.55 * directional)

    view_temporal = (
        both_festival
        * (
            0.40 * same_intent
            + 0.30 * same_audience
            + 0.15 * same_category
            + 0.15 * same_style
        )
    ) * (0.50 + 0.50 * directional)

    base_views = np.stack(
        [view_direct, view_structural, view_cross_channel, view_substitution, view_temporal],
        axis=-1,
    ).astype(np.float32)

    noise_train = rng.normal(scale=0.05, size=base_views.shape).astype(np.float32)
    noise_eval = rng.normal(scale=0.05, size=base_views.shape).astype(np.float32)
    drift = np.zeros_like(base_views, dtype=np.float32)
    drift[..., 3] += 0.08 * same_audience * diff_category  # substitution drift
    drift[..., 4] += 0.12 * both_festival                  # temporal drift

    train_view_scores = np.clip(base_views + noise_train, 0.0, None)
    eval_view_scores = np.clip(base_views + drift + noise_eval, 0.0, None)

    weights = np.array([0.34, 0.15, 0.14, 0.22, 0.15], dtype=np.float32)
    train_scores = np.sum(train_view_scores * weights[None, None, :], axis=-1)
    eval_scores = np.sum(eval_view_scores * weights[None, None, :], axis=-1)

    for mat in (train_scores, eval_scores):
        np.fill_diagonal(mat, 0.0)
    for idx in range(train_view_scores.shape[-1]):
        np.fill_diagonal(train_view_scores[..., idx], 0.0)
        np.fill_diagonal(eval_view_scores[..., idx], 0.0)

    # suppress a small portion of train edges to simulate emerging rivals in eval
    emergence = rng.random(size=train_scores.shape) < 0.12
    train_scores = np.where(emergence, train_scores * 0.72, train_scores).astype(np.float32)
    train_view_scores = np.where(emergence[..., None], train_view_scores * 0.72, train_view_scores).astype(np.float32)

    return ToyMarket(
        category=category,
        brand=brand,
        intent=intent,
        style=style,
        audience=audience,
        price=price,
        channel=channel,
        merchant_tier=merchant_tier,
        festival_active=festival_active,
        behavior=behavior,
        train_view_scores=train_view_scores,
        eval_view_scores=eval_view_scores,
        train_scores=train_scores.astype(np.float32),
        eval_scores=eval_scores.astype(np.float32),
    )


def build_retrieval_triplets(
    market: ToyMarket,
    negatives_per_positive: int = 4,
) -> list[tuple[int, int, int, np.ndarray]]:
    """Return (source, view, positive, negatives[]) training tuples."""
    n = market.train_scores.shape[0]
    tuples: list[tuple[int, int, int, np.ndarray]] = []
    rng = np.random.default_rng(123)
    for view_idx in range(len(VIEW_NAMES)):
        scores = market.train_view_scores[..., view_idx]
        for src in range(n):
            order = np.argsort(scores[src])[::-1]
            positives = [j for j in order if j != src and scores[src, j] > 0.22][:4]
            negatives_pool = [j for j in order[::-1] if j != src][:50]
            if not positives or not negatives_pool:
                continue
            for pos in positives:
                negs = rng.choice(
                    negatives_pool,
                    size=min(negatives_per_positive, len(negatives_pool)),
                    replace=False,
                )
                tuples.append((src, view_idx, int(pos), negs.astype(np.int64)))
    return tuples


def group_ground_truth(scores: np.ndarray, k: int = 20, threshold: float = 0.25) -> list[np.ndarray]:
    n = scores.shape[0]
    groups: list[np.ndarray] = []
    for i in range(n):
        order = np.argsort(scores[i])[::-1]
        keep = np.array([j for j in order if j != i and scores[i, j] > threshold][:k], dtype=np.int64)
        groups.append(keep)
    return groups


def relevance_labels(scores: np.ndarray, candidates: list[np.ndarray]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for src, cand in enumerate(candidates):
        out.append(scores[src, cand].astype(np.float32))
    return out


def build_profile_strings(market: ToyMarket, index: int) -> dict[str, str]:
    """Used only for README / debugging; keeps the AR/QR interface explicit."""
    base = {
        "f1": f"category={int(market.category[index])}; brand={int(market.brand[index])}",
        "f2": f"intent={int(market.intent[index])}",
        "f3": f"style={int(market.style[index])}",
        "f4": f"audience={int(market.audience[index])}",
        "f5": f"price_band={int(market.price[index])}",
        "f6": f"channel={int(market.channel[index])}; tier={int(market.merchant_tier[index])}",
    }
    return {
        "AR": " | ".join(base.values()),
        "direct_homogeneous": f"[direct] {base['f1']} | {base['f2']} | {base['f5']}",
        "intra_brand_structural": f"[structural] {base['f1']} | {base['f6']}",
        "intra_brand_cross_channel": f"[cross_channel] {base['f1']} | {base['f6']}",
        "cross_category_substitution": f"[substitution] {base['f2']} | {base['f3']} | {base['f4']} | {base['f5']}",
        "temporal_campaign": f"[temporal] {base['f2']} | {base['f4']} | festival={int(market.festival_active[index])}",
    }

