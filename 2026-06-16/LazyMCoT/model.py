"""LazyMCoT reference implementation (model side).

Faithfully implements the *logic* of the paper:
  - zero-cost first-token statistics (topp, delta_logit)
  - GBDT Adaptive Router with conformal (alpha-quantile) threshold calibration
  - Collaborative Grounding (Algorithm 1): entity decomposition, parallel
    attention-saliency branch + visual-expert branch, two-stage union/refine,
    Localized Panel Display (LPD), VLM re-query.

What is FAITHFUL vs STUBBED
---------------------------
* FAITHFUL: routing math, conformal calibration, attention->saliency (Gaussian
  smoothing, normalization, aggregation, thresholding -> boxes), two-stage
  union/refine with IoU coverage, LPD rendering, end-to-end orchestration.
* STUBBED (clean interfaces + documented pseudocode): the VLM and the visual
  expert (SAM3 / Grounding-DINO). The stubs are deterministic toy perceivers so
  the whole pipeline runs on CPU; in practice they wrap real backbones
  (e.g. Qwen2.5-VL for the VLM, SAM3 for the visual expert).
"""
from __future__ import annotations

import hashlib
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.ensemble import GradientBoostingClassifier

from data import COLOR_MATRIX, VOCAB, COLOR_RGB

Box = Tuple[float, float, float, float]

# Perception grids: the VLM "sees" the image at a coarse token grid (heavy
# downsampling, like a ViT patch grid), while the visual expert works at a finer
# grid. This gap is exactly why small targets need grounding.
VLM_GRID = 40          # VLM visual-token grid (N = VLM_GRID**2)
SAM_GRID = 72          # visual-expert resolution
COLOR_SIGMA = 0.22     # color-membership bandwidth (RGB in [0,1])
LOGIT_SCALE = 1.1      # evidence -> logit scale
NOISE_STD = 1.2        # deterministic perceptual noise


# ----------------------------------------------------------------------------
# small numpy helpers
# ----------------------------------------------------------------------------
def _img_grid(img: Image.Image, grid: int) -> np.ndarray:
    g = img.convert("RGB").resize((grid, grid), Image.BOX)
    return np.asarray(g, dtype=np.float64) / 255.0  # [grid,grid,3]


def _color_evidence(arr: np.ndarray) -> np.ndarray:
    """Soft per-color membership summed over pixels -> evidence vector [V].

    A saturation gate zeroes out gray pixels (background + distractors), so only
    genuinely colored object pixels contribute -> evidence tracks how much of the
    target color is *visible* at this resolution.
    """
    sat = arr.max(-1) - arr.min(-1)                              # [g,g]
    gate = np.clip((sat - 0.15) / 0.20, 0.0, 1.0)               # [g,g]
    diff = arr[:, :, None, :] - COLOR_MATRIX[None, None, :, :]   # [g,g,V,3]
    d2 = (diff ** 2).sum(-1)                                     # [g,g,V]
    member = np.exp(-d2 / (2 * COLOR_SIGMA ** 2))                # [g,g,V]
    return (member * gate[:, :, None]).sum(axis=(0, 1))         # [V]


def _saturation_grid(img: Image.Image, grid: int) -> np.ndarray:
    arr = _img_grid(img, grid)
    return arr.max(-1) - arr.min(-1)  # [grid,grid] in [0,1]


def _seeded_noise(img: Image.Image, n: int) -> np.ndarray:
    h = hashlib.md5(img.tobytes()).digest()
    seed = int.from_bytes(h[:4], "little")
    return np.random.default_rng(seed).normal(0.0, NOISE_STD, size=n)


def _gauss1d(sigma: float) -> np.ndarray:
    r = max(1, int(3 * sigma))
    x = np.arange(-r, r + 1)
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def gaussian_blur(a: np.ndarray, sigma: float) -> np.ndarray:
    k = _gauss1d(sigma)
    a = np.apply_along_axis(lambda m: np.convolve(m, k, "same"), 0, a)
    a = np.apply_along_axis(lambda m: np.convolve(m, k, "same"), 1, a)
    return a


def _components(mask: np.ndarray) -> List[Box]:
    """Connected-component bounding boxes (4-neighborhood) in grid coords."""
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    boxes: List[Box] = []
    for i in range(H):
        for j in range(W):
            if mask[i, j] and not seen[i, j]:
                stack = [(i, j)]
                seen[i, j] = True
                y0 = y1 = i
                x0 = x1 = j
                while stack:
                    y, x = stack.pop()
                    y0, y1 = min(y0, y), max(y1, y)
                    x0, x1 = min(x0, x), max(x1, x)
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
                boxes.append((x0, y0, x1 + 1, y1 + 1))
    return boxes


def _scale_boxes(boxes: Sequence[Box], grid: int, size: int) -> List[Box]:
    f = size / grid
    return [(b[0] * f, b[1] * f, b[2] * f, b[3] * f) for b in boxes]


def iou(a: Box, b: Box) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


# ----------------------------------------------------------------------------
# First-token statistics (zero cost, Eq. topp / delta_logit)
# ----------------------------------------------------------------------------
def compute_first_token_stats(logits: np.ndarray, option_ids: Sequence[int]) -> Tuple[float, float]:
    """topp     = max_{i in O} softmax(z)_i over option-restricted renormalization
       delta    = max_{i in O} z_i - max_{j not in O} z_j
    """
    logits = np.asarray(logits, dtype=np.float64)
    opt = np.asarray(option_ids, dtype=int)
    zo = logits[opt]
    p = np.exp(zo - zo.max())
    p = p / p.sum()
    topp = float(p.max())

    mask = np.ones(len(logits), dtype=bool)
    mask[opt] = False
    max_in = float(zo.max())
    max_out = float(logits[mask].max()) if mask.any() else 0.0
    return topp, max_in - max_out


# ----------------------------------------------------------------------------
# VLM stub
# ----------------------------------------------------------------------------
class VLMInterface:
    """Deterministic toy VLM. In practice wraps a real VLM (Qwen2.5-VL, ...).

    Real pseudocode for first_token_logits:
        inputs = processor(image, prompt_with_options)
        out = model(**inputs)              # single forward pass
        z = out.logits[0, -1]              # next-token logits over vocab V
        return z

    Here the toy perceiver downsamples to a coarse token grid and reads color
    evidence -> small/low-contrast targets vanish, giving low first-token
    confidence (exactly the "hard" regime targeted by routing).
    """

    def __init__(self):
        self.n_forward = 0

    def reset_counter(self):
        self.n_forward = 0

    def _read_logits(self, img: Image.Image, option_ids: Sequence[int], grid: int = VLM_GRID) -> np.ndarray:
        arr = _img_grid(img, grid)
        ev = _color_evidence(arr)                       # [V]
        logits = LOGIT_SCALE * ev + _seeded_noise(img, len(VOCAB))
        return logits

    # --- single forward pass: next-token logits over vocab -------------------
    def first_token_logits(self, image: Image.Image, question: str, option_ids: Sequence[int]) -> np.ndarray:
        self.n_forward += 1
        return self._read_logits(image, option_ids)

    # --- direct generation (argmax over options) -----------------------------
    def generate_answer(self, image: Image.Image, question: str, option_ids: Sequence[int]) -> int:
        self.n_forward += 1
        logits = self._read_logits(image, option_ids)
        return int(option_ids[int(np.argmax(np.asarray(logits)[list(option_ids)]))])

    # --- entity decomposition (text-only call) -------------------------------
    def decompose_entities(self, question: str) -> List[str]:
        """Real: prompt the VLM to list referred entities/nouns to localize.
        Toy: the question always refers to the small target object."""
        self.n_forward += 1
        return ["small target object"]

    # --- cross-modal attention A in R^{T x N} --------------------------------
    def cross_modal_attention(self, image: Image.Image, entities: List[str]) -> np.ndarray:
        """Return [T, N] attention from T entity text tokens to N visual tokens.
        Real: average selected layers/heads of text->image cross attention.
        Toy: each entity-token attends to salient (saturated) visual tokens."""
        self.n_forward += 1
        sal = _saturation_grid(image, VLM_GRID).reshape(-1)      # [N]
        sal = sal / (sal.max() + 1e-8)
        tokens = [w for e in entities for w in e.split()]
        T = max(1, len(tokens))
        rng = np.random.default_rng(int.from_bytes(hashlib.md5(image.tobytes()).digest()[4:8], "little"))
        A = np.stack([np.clip(sal + rng.normal(0, 0.03, sal.shape), 0, None) for _ in range(T)], 0)
        return A  # [T, N]

    # --- final answer from the Localized Panel Display -----------------------
    def answer_with_lpd(self, lpd_image: Image.Image, question: str, option_ids: Sequence[int]) -> int:
        """Re-query the VLM on the rendered panel (zoomed evidence) -> answer."""
        self.n_forward += 1
        logits = self._read_logits(lpd_image, option_ids)
        return int(option_ids[int(np.argmax(np.asarray(logits)[list(option_ids)]))])


# ----------------------------------------------------------------------------
# Visual expert stub (SAM3 / Grounding-DINO)
# ----------------------------------------------------------------------------
class VisualExpertSAM3:
    """Open-vocabulary detector stub.

    Real pseudocode:
        boxes = sam3.detect(image, text_prompts=entities)   # or Grounding-DINO
        return [b.xyxy for b in boxes if b.score > thr]

    Toy: detect saturated blobs (the colored targets) via connected components on
    a saturation map. Operating at SAM_GRID (finer than the VLM grid) lets it
    recover targets the VLM misses.
    """

    def __init__(self, sat_thr: float = 0.22, grid: int = SAM_GRID):
        self.sat_thr = sat_thr
        self.grid = grid

    def detect(self, image: Image.Image, entities: List[str]) -> List[Box]:
        sat = _saturation_grid(image, self.grid)
        mask = sat > self.sat_thr
        if not mask.any():
            return []
        gboxes = _components(mask)
        size = image.size[0]  # square canvas
        boxes = _scale_boxes(gboxes, self.grid, size)
        # drop degenerate boxes
        return [b for b in boxes if (b[2] - b[0]) >= 1 and (b[3] - b[1]) >= 1]


# ----------------------------------------------------------------------------
# Adaptive Router: GBDT + conformal threshold
# ----------------------------------------------------------------------------
class AdaptiveRouter:
    """g_theta = GBDT predicting p(hard / ori-wrong). s(x) = log(p/(1-p))."""

    def __init__(self, **gbdt_kwargs):
        params = dict(n_estimators=120, max_depth=3, learning_rate=0.1, random_state=0)
        params.update(gbdt_kwargs)
        self.model = GradientBoostingClassifier(**params)
        self.s_floor: float = -np.inf

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AdaptiveRouter":
        self.model.fit(X, y)
        return self

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def score(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        p = self.model.predict_proba(X)[:, 1]
        return self._logit(p)

    def calibrate_threshold(self, scores_mr: np.ndarray, alpha: float) -> float:
        """Conformal lower bound: s_floor = Q_alpha({s_i}) over the must-recall
        (ori-wrong) subset. ~(1-alpha) of hard samples score >= s_floor, i.e. a
        controllable hard-sample recall. Smaller alpha => lower floor => route
        more => more conservative."""
        scores_mr = np.asarray(scores_mr, dtype=np.float64)
        self.s_floor = float(np.quantile(scores_mr, alpha))
        return self.s_floor

    def decide(self, x) -> bool:
        """True => route to Collaborative Grounding; False => Direct."""
        return bool(self.score(x)[0] >= self.s_floor)


# ----------------------------------------------------------------------------
# Collaborative Grounding (Algorithm 1)
# ----------------------------------------------------------------------------
class CollaborativeGrounding:
    def __init__(self, vlm: VLMInterface, expert: VisualExpertSAM3,
                 sigma: float = 1.2, tau: float = 0.45, cov_thr: float = 0.25,
                 enlarge: float = 1.8):
        self.vlm = vlm
        self.expert = expert
        self.sigma = sigma          # Gaussian bandwidth
        self.tau = tau              # saliency threshold
        self.cov_thr = cov_thr      # IoU coverage threshold
        self.enlarge = enlarge      # stage-2 crop enlargement

    # attention -> saliency -> boxes (B_att)
    def _attention_boxes(self, image: Image.Image, A: np.ndarray) -> Tuple[List[Box], np.ndarray]:
        T = A.shape[0]
        g = VLM_GRID
        maps = []
        for t in range(T):
            m = A[t].reshape(g, g)
            m = gaussian_blur(m, self.sigma)            # smooth
            m = m / (m.max() + 1e-8)                     # normalize per token
            maps.append(m)
        sal = np.mean(maps, axis=0)                      # aggregate across tokens
        sal = sal / (sal.max() + 1e-8)
        mask = sal > self.tau
        gboxes = _components(mask)
        boxes = _scale_boxes(gboxes, g, image.size[0])
        return boxes, sal

    def _covered(self, b: Box, ref: List[Box]) -> bool:
        return any(iou(b, r) >= self.cov_thr for r in ref)

    def _crop_box(self, image: Image.Image, b: Box) -> Tuple[Image.Image, Box]:
        S = image.size[0]
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        w = (b[2] - b[0]) * self.enlarge
        h = (b[3] - b[1]) * self.enlarge
        x0 = max(0, int(cx - w / 2)); y0 = max(0, int(cy - h / 2))
        x1 = min(S, int(cx + w / 2)); y1 = min(S, int(cy + h / 2))
        if x1 <= x0: x1 = min(S, x0 + 4)
        if y1 <= y0: y1 = min(S, y0 + 4)
        return image.crop((x0, y0, x1, y1)), (x0, y0, x1, y1)

    def run(self, image: Image.Image, question: str, option_ids: Sequence[int]) -> int:
        # 1) decompose question into entities
        entities = self.vlm.decompose_entities(question)

        # 2) two parallel branches
        b_exp = self.expert.detect(image, entities)               # visual expert
        A = self.vlm.cross_modal_attention(image, entities)       # [T, N]
        b_att, _sal = self._attention_boxes(image, A)             # attention branch

        # 3) stage-1 union
        B1 = list(b_exp) + list(b_att)

        # 4) stage-2 refine: re-query the expert on enlarged crops of attention
        #    boxes not covered by the expert; map deltas back to image coords.
        delta: List[Box] = []
        for b in b_att:
            if self._covered(b, b_exp):
                continue
            crop, (ox, oy, _, _) = self._crop_box(image, b)
            for db in self.expert.detect(crop, entities):
                delta.append((db[0] + ox, db[1] + oy, db[2] + ox, db[3] + oy))
        B2 = B1 + delta
        if not B2:                                                # safety fallback
            B2 = [(0, 0, image.size[0], image.size[1])]

        # 5) render Localized Panel Display and re-query VLM
        lpd = render_lpd(image, B2)
        return self.vlm.answer_with_lpd(lpd, question, option_ids)


# ----------------------------------------------------------------------------
# Localized Panel Display (LPD)
# ----------------------------------------------------------------------------
_BORDER = [(255, 0, 0), (0, 180, 0), (0, 0, 255), (255, 140, 0), (160, 0, 200), (0, 170, 170)]


def render_lpd(image: Image.Image, boxes: Sequence[Box], crop_px: int = 120) -> Image.Image:
    """Draw colored borders + legends on the image and append zoomed crops of
    each evidence box (a panel display the VLM can re-read at higher effective
    resolution)."""
    S = image.size[0]
    ann = image.convert("RGB").copy()
    d = ImageDraw.Draw(ann)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    # dedup near-identical boxes, keep up to 6
    kept: List[Box] = []
    for b in boxes:
        if all(iou(b, k) < 0.85 for k in kept):
            kept.append(b)
    kept = kept[:6]

    crops = []
    for i, b in enumerate(kept):
        col = _BORDER[i % len(_BORDER)]
        d.rectangle([b[0], b[1], b[2], b[3]], outline=col, width=3)
        lbl = f"#{i+1}"
        if font is not None:
            d.text((b[0] + 2, max(0, b[1] - 12)), lbl, fill=col, font=font)
        crop = image.crop((int(b[0]), int(b[1]), int(b[2]), int(b[3]))).resize((crop_px, crop_px), Image.BICUBIC)
        cc = ImageDraw.Draw(crop)
        cc.rectangle([0, 0, crop_px - 1, crop_px - 1], outline=col, width=4)
        if font is not None:
            cc.text((3, 3), lbl, fill=col, font=font)
        crops.append(crop)

    panel_w = max(S, crop_px * max(1, len(crops)))
    panel_h = S + crop_px + 16
    panel = Image.new("RGB", (panel_w, panel_h), (255, 255, 255))
    panel.paste(ann, (0, 0))
    for i, c in enumerate(crops):
        panel.paste(c, (i * crop_px, S + 12))
    return panel


# ----------------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------------
class LazyMCoT:
    def __init__(self, vlm: VLMInterface, router: AdaptiveRouter, cg: CollaborativeGrounding):
        self.vlm = vlm
        self.router = router
        self.cg = cg

    def predict(self, sample) -> dict:
        self.vlm.reset_counter()
        logits = self.vlm.first_token_logits(sample.image, sample.question, sample.option_ids)
        topp, delta = compute_first_token_stats(logits, sample.option_ids)
        direct_pred = int(sample.option_ids[int(np.argmax(np.asarray(logits)[list(sample.option_ids)]))])

        route = self.router.decide([[topp, delta]])  # True => CG
        if not route:
            pred, mode = direct_pred, "direct"
        else:
            pred = self.cg.run(sample.image, sample.question, sample.option_ids)
            mode = "grounded"

        return {
            "pred": pred,
            "mode": mode,
            "topp": topp,
            "delta_logit": delta,
            "score": float(self.router.score([[topp, delta]])[0]),
            "n_forward": self.vlm.n_forward,
            "direct_pred": direct_pred,
        }
