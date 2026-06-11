from dataclasses import dataclass

import numpy as np


def set_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


@dataclass(frozen=True)
class ToyVideoConfig:
    num_base: int = 800
    num_dups: int = 400
    frames: int = 32
    feat_dim: int = 64
    pos_pairs: int = 1000
    neg_pairs: int = 1000


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def generate_videos(cfg: ToyVideoConfig, seed: int = 7):
    rng = set_seed(seed)

    base = rng.normal(size=(cfg.num_base, cfg.frames, cfg.feat_dim)).astype(np.float32)
    base = _normalize(base)

    # duplicates: copy a random segment from a base video, add small noise & simple edit
    dups = np.empty((cfg.num_dups, cfg.frames, cfg.feat_dim), dtype=np.float32)
    dup_meta = []

    for i in range(cfg.num_dups):
        src = int(rng.integers(0, cfg.num_base))
        seg_len = int(rng.integers(cfg.frames // 3, cfg.frames // 2 + 1))
        src_start = int(rng.integers(0, cfg.frames - seg_len + 1))

        seg = base[src, src_start : src_start + seg_len].copy()

        # simulate lightweight edits
        seg = seg + rng.normal(scale=0.05, size=seg.shape).astype(np.float32)
        seg = _normalize(seg)

        out = rng.normal(size=(cfg.frames, cfg.feat_dim)).astype(np.float32)
        out = _normalize(out)

        dst_start = int(rng.integers(0, cfg.frames - seg_len + 1))
        out[dst_start : dst_start + seg_len] = seg

        # watermark-like bias
        out = out + 0.01
        out = _normalize(out)

        dups[i] = out
        dup_meta.append({"dup_id": cfg.num_base + i, "src_id": src, "src_range": (src_start, src_start + seg_len), "dup_range": (dst_start, dst_start + seg_len)})

    videos = np.concatenate([base, dups], axis=0)

    return videos, dup_meta


def build_pairs(videos: np.ndarray, dup_meta: list[dict], cfg: ToyVideoConfig, seed: int = 7):
    rng = set_seed(seed + 1)

    # positive pairs are (dup, src)
    pos = []
    for _ in range(cfg.pos_pairs):
        m = dup_meta[int(rng.integers(0, len(dup_meta)))]
        dup_id = int(m["dup_id"])
        src_id = int(m["src_id"])
        src_start, src_end = m["src_range"]
        dup_start, dup_end = m["dup_range"]
        overlap_ratio = (dup_end - dup_start) / cfg.frames
        pos.append({"a": dup_id, "b": src_id, "label": 1, "overlap_ratio": float(overlap_ratio), "meta": m})

    # negative pairs random
    total = videos.shape[0]
    neg = []
    for _ in range(cfg.neg_pairs):
        a = int(rng.integers(0, total))
        b = int(rng.integers(0, total))
        if a == b:
            continue
        neg.append({"a": a, "b": b, "label": 0, "overlap_ratio": 0.0, "meta": None})

    pairs = pos + neg
    rng.shuffle(pairs)
    return pairs
