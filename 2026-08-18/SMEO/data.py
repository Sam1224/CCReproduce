import math
import random
from dataclasses import dataclass
from typing import List, Sequence

import torch
from torch.utils.data import Dataset


MODALITIES = ["hero_image", "detail_image", "ugc_video", "3d_spin"]
MODALITY_VECTORS = {
    "hero_image": [1.0, 0.0, 0.0, 0.0],
    "detail_image": [0.0, 1.0, 0.0, 0.0],
    "ugc_video": [0.0, 0.0, 1.0, 0.0],
    "3d_spin": [0.0, 0.0, 0.0, 1.0],
}


@dataclass
class Session:
    assets: List[List[float]]
    purchase_goal: List[float]
    best_order: List[int]
    baseline_order: List[int]


class PrefixUtilityDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        x, y = self.rows[idx]
        return {
            "x": torch.tensor(x, dtype=torch.float32),
            "y": torch.tensor([y], dtype=torch.float32),
        }


class RankingDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        x, y = self.rows[idx]
        return {
            "x": torch.tensor(x, dtype=torch.float32),
            "y": torch.tensor(y, dtype=torch.long),
        }


def asset_dim() -> int:
    return 11


def rank_asset_dim() -> int:
    return 12


def state_dim() -> int:
    return 17


def rank_input_dim(num_assets: int = 6) -> int:
    return state_dim() + rank_asset_dim() * num_assets


def _asset_features(rng: random.Random, modality: str, goal: Sequence[float]) -> List[float]:
    evidence = [
        rng.uniform(0.2, 1.0),
        rng.uniform(0.1, 1.0),
        rng.uniform(0.1, 1.0),
    ]
    friction = rng.uniform(0.05, 0.35)
    alignment = sum(evidence[i] * goal[i] for i in range(3)) / 3.0
    style = rng.uniform(0.1, 0.9)
    freshness = rng.uniform(0.1, 1.0)
    return MODALITY_VECTORS[modality] + evidence + [friction, alignment, style, freshness]


def _marginal_gain(asset: Sequence[float], goal: Sequence[float], seen_modalities: set[str]) -> float:
    evidence = asset[4:7]
    friction = asset[7]
    alignment = asset[8]
    freshness = asset[10]
    modality = MODALITIES[asset[:4].index(max(asset[:4]))]
    novelty = 0.12 if modality not in seen_modalities else 0.0
    return float(sum(evidence[i] * goal[i] for i in range(3)) + 0.4 * alignment + 0.15 * freshness + novelty - 0.55 * friction)


def _prefix_state(assets: List[List[float]], prefix: List[int], goal: Sequence[float]) -> List[float]:
    if prefix:
        viewed = [assets[i] for i in prefix]
        avg_asset = [sum(row[k] for row in viewed) / len(viewed) for k in range(asset_dim())]
        gained = sum(max(0.0, row[8] - row[7]) for row in viewed)
    else:
        avg_asset = [0.0] * asset_dim()
        gained = 0.0
    utility = 1.0 - math.exp(-max(0.0, gained))
    return list(goal) + [len(prefix) / len(assets), utility, gained] + avg_asset


def _utility_from_prefix(assets: List[List[float]], prefix: List[int]) -> float:
    gained = sum(max(0.0, assets[i][8] - assets[i][7]) for i in prefix)
    return 1.0 - math.exp(-gained)


def build_sessions(num_sessions: int, seed: int, num_assets: int = 6):
    rng = random.Random(seed)
    sessions: List[Session] = []
    for _ in range(num_sessions):
        goal = [rng.uniform(0.2, 1.0), rng.uniform(0.2, 1.0), rng.uniform(0.2, 1.0)]
        assets = []
        for idx in range(num_assets):
            modality = MODALITIES[idx % len(MODALITIES)]
            assets.append(_asset_features(rng, modality, goal))
        seen_modalities = set()
        ranked = []
        remaining = list(range(num_assets))
        while remaining:
            best = max(remaining, key=lambda i: _marginal_gain(assets[i], goal, seen_modalities))
            ranked.append(best)
            modality = MODALITIES[assets[best][:4].index(max(assets[best][:4]))]
            seen_modalities.add(modality)
            remaining.remove(best)
        baseline = sorted(range(num_assets), key=lambda i: assets[i][9], reverse=True)
        sessions.append(Session(assets=assets, purchase_goal=goal, best_order=ranked, baseline_order=baseline))
    return sessions


def build_utility_rows(sessions: List[Session]):
    rows = []
    for session in sessions:
        for prefix_len in range(len(session.assets)):
            prefix = session.best_order[:prefix_len]
            state = _prefix_state(session.assets, prefix, session.purchase_goal)
            target = _utility_from_prefix(session.assets, prefix)
            rows.append((state, target))
    return rows


def _flatten_rank_row(state: Sequence[float], assets: List[List[float]]) -> List[float]:
    flat = list(state)
    for asset in assets:
        flat.extend(asset)
    return flat


def build_rank_rows(sessions: List[Session]):
    rows = []
    for session in sessions:
        for prefix_len in range(len(session.assets) - 1):
            prefix = session.best_order[:prefix_len]
            state = _prefix_state(session.assets, prefix, session.purchase_goal)
            remaining = [i for i in range(len(session.assets)) if i not in prefix]
            sorted_remaining = remaining + [i for i in range(len(session.assets)) if i in prefix]
            flattened_assets = []
            for idx in sorted_remaining:
                asset = session.assets[idx]
                mask = [0.0] if idx in prefix else [1.0]
                flattened_assets.append(asset + mask)
            target_original = session.best_order[prefix_len]
            target = sorted_remaining.index(target_original)
            rows.append((_flatten_rank_row(state, flattened_assets), target))
    return rows


def make_splits():
    train_sessions = build_sessions(240, seed=11)
    val_sessions = build_sessions(60, seed=23)
    test_sessions = build_sessions(80, seed=37)
    return {
        "train_sessions": train_sessions,
        "val_sessions": val_sessions,
        "test_sessions": test_sessions,
        "utility_train": PrefixUtilityDataset(build_utility_rows(train_sessions)),
        "utility_val": PrefixUtilityDataset(build_utility_rows(val_sessions)),
        "rank_train": RankingDataset(build_rank_rows(train_sessions)),
        "rank_val": RankingDataset(build_rank_rows(val_sessions)),
    }


def decision_swipes(order: List[int], assets: List[List[float]], threshold: float = 0.82) -> int:
    utility = 0.0
    for step, idx in enumerate(order, start=1):
        utility = 1.0 - math.exp(-sum(max(0.0, assets[j][8] - assets[j][7]) for j in order[:step]))
        if utility >= threshold:
            return step
    return len(order)
