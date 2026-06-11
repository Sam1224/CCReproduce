import numpy as np


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


class ToyMultiLevelEncoder:
    """A lightweight stand-in for ML-VE.

    Input video is already a sequence of frame features. We simulate:
    - frame-level embeddings: per-frame normalized vectors
    - clip-level embedding: coarse pooled representation for retrieval
    """

    def __init__(self, clip_stride: int = 4):
        self.clip_stride = clip_stride

    def encode_frame(self, video: np.ndarray) -> np.ndarray:
        return _normalize(video.astype(np.float32))

    def encode_clip(self, video: np.ndarray) -> np.ndarray:
        v = video.astype(np.float32)
        chunks = []
        for i in range(0, v.shape[0], self.clip_stride):
            chunks.append(v[i : i + self.clip_stride].mean(axis=0))
        emb = np.stack(chunks, axis=0).mean(axis=0)
        return _normalize(emb)


class ToyDiFSiM:
    """A simplified DiF-SiM matcher.

    - Uses differential features to highlight temporal changes.
    - Searches diagonal offsets to localize the most plausible copied segment.
    """

    def __init__(self, diff_weight: float = 0.5, cell_thr: float = 0.65):
        self.diff_weight = diff_weight
        self.cell_thr = cell_thr

    def _diff(self, x: np.ndarray) -> np.ndarray:
        d = np.zeros_like(x)
        d[1:] = x[1:] - x[:-1]
        return _normalize(d)

    def match(self, fa: np.ndarray, fb: np.ndarray) -> dict:
        fa = _normalize(fa)
        fb = _normalize(fb)
        da = self._diff(fa)
        db = self._diff(fb)

        s0 = fa @ fb.T
        sd = da @ db.T
        sim = s0 + self.diff_weight * sd

        l = fa.shape[0]

        best = {"score": -1.0, "overlap_len": 0, "offset": 0}

        # offset = j - i
        for offset in range(-l + 1, l):
            i0 = max(0, -offset)
            j0 = max(0, offset)
            diag_len = l - max(i0, j0)
            if diag_len < 4:
                continue

            diag = np.array([sim[i0 + t, j0 + t] for t in range(diag_len)], dtype=np.float32)

            # longest contiguous run where similarity is high
            mask = diag > self.cell_thr
            cur = 0
            best_run = 0
            best_run_score = 0.0
            best_run_start = 0
            run_sum = 0.0

            for t in range(diag_len):
                if mask[t]:
                    cur += 1
                    run_sum += float(diag[t])
                    if cur > best_run:
                        best_run = cur
                        best_run_start = t - cur + 1
                        best_run_score = run_sum / cur
                else:
                    cur = 0
                    run_sum = 0.0

            if best_run == 0:
                # fall back to mean diag similarity
                best_run = 0
                best_run_score = float(diag.mean())

            if best_run_score > best["score"]:
                best = {"score": best_run_score, "overlap_len": best_run, "offset": offset, "start": best_run_start}

        overlap_ratio = best["overlap_len"] / l
        return {
            "score": float(best["score"]),
            "overlap_ratio": float(overlap_ratio),
            "offset": int(best["offset"]),
        }


def cosine_topk(query: np.ndarray, keys: np.ndarray, k: int = 5) -> np.ndarray:
    query = _normalize(query)
    keys = _normalize(keys)
    scores = keys @ query
    idx = np.argsort(-scores)[:k]
    return idx
