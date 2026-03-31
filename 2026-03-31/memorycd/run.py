from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def bleu1(reference: str, hypothesis: str) -> float:
    ref = [t for t in reference.lower().split() if t]
    hyp = [t for t in hypothesis.lower().split() if t]
    if not hyp:
        return 0.0
    ref_counts = {}
    for t in ref:
        ref_counts[t] = ref_counts.get(t, 0) + 1

    hit = 0
    used = {}
    for t in hyp:
        used[t] = used.get(t, 0) + 1
        if used[t] <= ref_counts.get(t, 0):
            hit += 1
    return hit / len(hyp)


def rouge_l(reference: str, hypothesis: str) -> float:
    ref = [t for t in reference.lower().split() if t]
    hyp = [t for t in hypothesis.lower().split() if t]
    if not ref or not hyp:
        return 0.0

    # LCS
    dp = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs = dp[-1][-1]
    prec = lcs / len(hyp)
    rec = lcs / len(ref)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def ndcg_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    order = np.argsort(-scores)[:k]
    gains = labels[order]
    discounts = 1.0 / np.log2(np.arange(2, gains.size + 2))
    dcg = float((gains * discounts).sum())
    ideal = np.sort(labels)[::-1][:k]
    idcg = float((ideal * discounts[: ideal.size]).sum())
    return dcg / (idcg + 1e-9)


@dataclass
class Interaction:
    domain: str
    item_id: int
    rating: float
    title: str
    review: str
    item_vec: np.ndarray


def gen_user_history(domains: List[str], *, history: int, d: int) -> List[Interaction]:
    hist: List[Interaction] = []
    user_pref = np.random.randn(d)

    for _ in range(history):
        dom = random.choice(domains)
        item_id = int(np.random.randint(0, 5000))
        item_vec = np.random.randn(d)
        score = float(np.dot(user_pref, item_vec) / (np.linalg.norm(user_pref) * np.linalg.norm(item_vec) + 1e-9))
        rating = float(np.clip(3.0 + score * 1.2 + np.random.randn() * 0.2, 1.0, 5.0))
        title = f"{dom} item {item_id} {'great' if rating > 3.5 else 'ok'}"
        review = f"I think this {dom} item {item_id} is {'excellent' if rating > 4 else 'decent'} and fits my taste ."
        hist.append(Interaction(dom, item_id, rating, title, review, item_vec))

    return hist


def split_memory(history: List[Interaction], *, eval_domain: str, cross_domain: bool) -> Tuple[List[Interaction], List[Interaction]]:
    # memory: past interactions, eval: recent interactions in eval_domain
    eval_items = [x for x in history[-20:] if x.domain == eval_domain]
    mem_items = history[:-20]

    if not cross_domain:
        mem_items = [x for x in mem_items if x.domain == eval_domain]

    return mem_items, eval_items


# ---------------- memory baselines ----------------

def baseline_rating_predict(mem: List[Interaction], target_item: Interaction) -> float:
    if not mem:
        return 3.0
    # domain-conditioned mean + similarity
    dom = [x.rating for x in mem if x.domain == target_item.domain]
    mu = float(np.mean(dom) if dom else np.mean([x.rating for x in mem]))
    return float(np.clip(mu, 1.0, 5.0))


def baseline_rank(mem: List[Interaction], candidates: List[Interaction]) -> np.ndarray:
    if not mem:
        return np.zeros(len(candidates))
    pref = np.mean([x.item_vec for x in mem], axis=0)
    pref = pref / (np.linalg.norm(pref) + 1e-9)
    scores = []
    for c in candidates:
        v = c.item_vec / (np.linalg.norm(c.item_vec) + 1e-9)
        scores.append(float(pref @ v))
    return np.asarray(scores)


def baseline_summarize(mem: List[Interaction], target_item: Interaction) -> str:
    # mimic user's title style by using most frequent sentiment token
    if not mem:
        tone = "good"
    else:
        words = " ".join([x.title for x in mem]).lower().split()
        tone = "great" if words.count("great") >= words.count("ok") else "ok"
    return f"{target_item.domain} item {target_item.item_id} {tone}"


def baseline_generate(mem: List[Interaction], target_item: Interaction) -> str:
    # mimic user's review style by using avg length and sentiment
    if not mem:
        sentiment = "decent"
    else:
        avg = float(np.mean([x.rating for x in mem]))
        sentiment = "excellent" if avg > 3.7 else "decent"
    return f"This {target_item.domain} item {target_item.item_id} is {sentiment} and matches my preferences ."


def run_user(domains: List[str], *, history: int, d: int, cross_domain: bool) -> Dict[str, float]:
    hist = gen_user_history(domains, history=history, d=d)
    eval_domain = random.choice(domains)
    mem, eval_items = split_memory(hist, eval_domain=eval_domain, cross_domain=cross_domain)
    if not eval_items:
        return {}

    # rating
    y = []
    p = []

    # ranking
    ndcg3 = []

    # text
    r_rouge = []
    r_bleu = []
    g_rouge = []
    g_bleu = []

    for tgt in eval_items[:10]:
        pred_rating = baseline_rating_predict(mem, tgt)
        y.append(tgt.rating)
        p.append(pred_rating)

        # ranking candidates: 1 positive (tgt) + negatives
        candidates = [tgt]
        for _ in range(9):
            candidates.append(random.choice(hist))
        labels = np.asarray([1.0] + [0.0] * 9)
        scores = baseline_rank(mem, candidates)
        ndcg3.append(ndcg_at_k(scores, labels, 3))

        # summarization
        hyp_t = baseline_summarize(mem, tgt)
        r_rouge.append(rouge_l(tgt.title, hyp_t))
        r_bleu.append(bleu1(tgt.title, hyp_t))

        # generation
        hyp_r = baseline_generate(mem, tgt)
        g_rouge.append(rouge_l(tgt.review, hyp_r))
        g_bleu.append(bleu1(tgt.review, hyp_r))

    y = np.asarray(y)
    p = np.asarray(p)
    mae = float(np.mean(np.abs(y - p)))
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))

    return {
        "mae": mae,
        "rmse": rmse,
        "ndcg@3": float(np.mean(ndcg3)),
        "summ_rougeL": float(np.mean(r_rouge)),
        "summ_bleu1": float(np.mean(r_bleu)),
        "gen_rougeL": float(np.mean(g_rouge)),
        "gen_bleu1": float(np.mean(g_bleu)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=200)
    ap.add_argument("--history", type=int, default=200)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    seed_everything(args.seed)

    domains = [
        "Books",
        "Electronics",
        "HomeKitchen",
        "PersonalCare",
        "Movies",
        "Sports",
        "Toys",
        "Music",
        "Beauty",
        "Grocery",
        "Software",
        "Automotive",
    ]

    def agg(rows: List[Dict[str, float]]) -> Dict[str, float]:
        keys = sorted({k for r in rows for k in r.keys()})
        out = {}
        for k in keys:
            out[k] = float(np.mean([r[k] for r in rows if k in r]))
        return out

    single = []
    cross = []

    for _ in range(args.users):
        single.append(run_user(domains, history=args.history, d=args.d, cross_domain=False))
        cross.append(run_user(domains, history=args.history, d=args.d, cross_domain=True))

    print({"setting": "single_domain", **agg(single)})
    print({"setting": "cross_domain", **agg(cross)})


if __name__ == "__main__":
    main()
