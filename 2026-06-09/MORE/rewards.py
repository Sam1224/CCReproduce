from __future__ import annotations

from dataclasses import dataclass


def correctness_reward(*, credit_limit: int, eligible: bool, text: str) -> float:
    limit_ok = str(credit_limit) in text
    if eligible:
        elig_ok = ("支持分期" in text) or ("可以分期" in text) or ("满足分期" in text)
    else:
        elig_ok = ("不支持分期" in text) or ("无法分期" in text) or ("不满足分期" in text)
    return 1.0 if (limit_ok and elig_ok) else 0.0


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def bleu2(reference: str, candidate: str) -> float:
    # char-level BLEU-2
    ref = list(reference)
    cand = list(candidate)

    p1 = _prec(ref, cand, 1)
    p2 = _prec(ref, cand, 2)

    if p1 == 0 or p2 == 0:
        return 0.0

    import math

    bp = 1.0
    if len(cand) < len(ref) and len(cand) > 0:
        bp = math.exp(1 - len(ref) / len(cand))

    return float(bp * math.exp(0.5 * (math.log(p1) + math.log(p2))))


def _prec(ref: list[str], cand: list[str], n: int) -> float:
    ref_ng = _ngrams(ref, n)
    cand_ng = _ngrams(cand, n)
    if not cand_ng:
        return 0.0

    from collections import Counter

    ref_c = Counter(ref_ng)
    cand_c = Counter(cand_ng)

    overlap = 0
    for g, c in cand_c.items():
        overlap += min(c, ref_c.get(g, 0))

    return overlap / max(len(cand_ng), 1)


def length_reward(text: str, *, target_len: int = 22) -> float:
    # reward closer to a reasonable length (avoid too short / too long)
    l = len(text)
    diff = abs(l - target_len)
    return max(0.0, 1.0 - diff / float(target_len))


def unique_trigram_ratio(text: str) -> float:
    tokens = list(text)
    tri = _ngrams(tokens, 3)
    if not tri:
        return 0.0
    return len(set(tri)) / len(tri)


@dataclass
class AdaptiveWeights:
    w_bleu: float = 1.0
    w_len: float = 0.5

    ema_bleu: float = 0.0
    ema_len: float = 0.0
    ema_var_bleu: float = 1.0
    ema_var_len: float = 1.0

    def update(self, *, bleu_r: float, len_r: float, alpha: float = 0.05) -> None:
        # moving mean
        self.ema_bleu = (1 - alpha) * self.ema_bleu + alpha * bleu_r
        self.ema_len = (1 - alpha) * self.ema_len + alpha * len_r

        # moving variance proxy
        self.ema_var_bleu = (1 - alpha) * self.ema_var_bleu + alpha * (bleu_r - self.ema_bleu) ** 2
        self.ema_var_len = (1 - alpha) * self.ema_var_len + alpha * (len_r - self.ema_len) ** 2

        inv_bleu = 1.0 / (self.ema_var_bleu + 1e-6)
        inv_len = 1.0 / (self.ema_var_len + 1e-6)
        s = inv_bleu + inv_len
        self.w_bleu = float(inv_bleu / s)
        self.w_len = float(inv_len / s)
