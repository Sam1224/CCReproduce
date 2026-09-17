from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch


@dataclass
class DataConfig:
    n_products: int = 2400
    n_queries: int = 900
    topk: int = 30
    seed: int = 7


@dataclass(frozen=True)
class Product:
    pid: str
    title: str
    attrs: Dict[str, str]


@dataclass(frozen=True)
class Query:
    qid: str
    text: str
    intent: Dict[str, str]


@dataclass
class AnnotatedPair:
    qid: str
    pid: str
    true_grade: int
    pseudo_grade: int
    confidence: float
    bucket: str


_BRANDS = ["acme", "freshco", "greenfarm", "snacklab", "vita", "yummy"]
_CATS = ["coffee", "tea", "snacks", "cereal", "sauce", "noodles"]
_FLAVORS = ["vanilla", "chocolate", "spicy", "sweet", "original", "lemon"]
_SIZES = ["small", "medium", "large"]
_DIET = ["regular", "gluten_free", "lactose_free", "vegan"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _tok(s: str) -> List[str]:
    return [t for t in s.lower().replace("-", " ").split() if t]


def build_vocab(texts: Iterable[str], min_freq: int = 1) -> Dict[str, int]:
    freq: Dict[str, int] = {}
    for t in texts:
        for w in _tok(t):
            freq[w] = freq.get(w, 0) + 1
    vocab = {"<pad>": 0, "<unk>": 1}
    for w, c in sorted(freq.items(), key=lambda x: (-x[1], x[0])):
        if c >= min_freq and w not in vocab:
            vocab[w] = len(vocab)
    return vocab


def encode(text: str, vocab: Dict[str, int]) -> List[int]:
    ids = [vocab.get(w, vocab["<unk>"]) for w in _tok(text)]
    return ids or [vocab["<unk>"]]


def make_toy_catalog(n_products: int, seed: int = 0) -> List[Product]:
    set_seed(seed)
    products: List[Product] = []
    for i in range(n_products):
        brand = random.choice(_BRANDS)
        cat = random.choice(_CATS)
        flavor = random.choice(_FLAVORS)
        size = random.choice(_SIZES)
        diet = random.choice(_DIET)
        title = f"{brand} {cat} {flavor} {size} {diet}".replace("_", " ")
        attrs = {"brand": brand, "cat": cat, "flavor": flavor, "size": size, "diet": diet}
        products.append(Product(pid=f"p{i:05d}", title=title, attrs=attrs))
    return products


def make_toy_queries(n_queries: int, seed: int = 0) -> List[Query]:
    set_seed(seed)
    queries: List[Query] = []
    for i in range(n_queries):
        brand = random.choice(_BRANDS)
        cat = random.choice(_CATS)
        intent: Dict[str, str] = {"brand": brand, "cat": cat}
        if random.random() < 0.55:
            intent["flavor"] = random.choice(_FLAVORS)
        if random.random() < 0.35:
            intent["diet"] = random.choice(_DIET)
        if random.random() < 0.20:
            intent["size"] = random.choice(_SIZES)
        text = " ".join([brand, cat] + [intent[k].replace("_", " ") for k in ["flavor", "diet", "size"] if k in intent])
        queries.append(Query(qid=f"q{i:05d}", text=text, intent=intent))
    return queries


def relevance_grade(q: Query, p: Product) -> int:
    score = 0
    for k, v in q.intent.items():
        if p.attrs.get(k) == v:
            score += 1
    if p.attrs.get("brand") != q.intent.get("brand") or p.attrs.get("cat") != q.intent.get("cat"):
        return 0
    grade = min(4, 2 + max(0, score - 2))
    if "diet" in q.intent and p.attrs.get("diet") != q.intent["diet"]:
        grade = max(1, grade - 1)
    return int(grade)


def train_valid_test_split(items: Sequence, ratios=(0.8, 0.1, 0.1), seed: int = 0):
    assert math.isclose(sum(ratios), 1.0)
    idx = list(range(len(items)))
    random.Random(seed).shuffle(idx)
    n = len(items)
    n_train = int(n * ratios[0])
    n_valid = int(n * ratios[1])
    train = [items[i] for i in idx[:n_train]]
    valid = [items[i] for i in idx[n_train : n_train + n_valid]]
    test = [items[i] for i in idx[n_train + n_valid :]]
    return train, valid, test


def build_corpus(queries: Sequence[Query], products: Sequence[Product]) -> Tuple[List[str], List[str]]:
    return [q.text for q in queries], [p.title for p in products]


def lexical_score(q: Query, p: Product) -> float:
    qt = set(_tok(q.text))
    pt = set(_tok(p.title))
    if not qt or not pt:
        return 0.0
    return len(qt & pt) / len(qt | pt)


def dense_baseline_score(q: Query, p: Product, proj: Dict[str, np.ndarray]) -> float:
    qv = np.zeros_like(next(iter(proj.values())))
    dv = np.zeros_like(qv)
    for t in _tok(q.text):
        if t in proj:
            qv += proj[t]
    for t in _tok(p.title):
        if t in proj:
            dv += proj[t]
    qn = np.linalg.norm(qv) + 1e-9
    dn = np.linalg.norm(dv) + 1e-9
    return float(np.dot(qv, dv) / (qn * dn))


def build_token_projection(queries: Sequence[Query], products: Sequence[Product], dim: int = 48, seed: int = 0):
    rng = np.random.default_rng(seed)
    tokens = set()
    for q in queries:
        tokens.update(_tok(q.text))
    for p in products:
        tokens.update(_tok(p.title))
    proj: Dict[str, np.ndarray] = {}
    for t in sorted(tokens):
        v = rng.normal(0, 1.0, size=(dim,)).astype(np.float32)
        proj[t] = v / (np.linalg.norm(v) + 1e-9)
    return proj


def topk_indices(scores: Sequence[float], k: int) -> List[int]:
    if k >= len(scores):
        return list(np.argsort(-np.array(scores)))
    return list(np.argpartition(-np.array(scores), k)[:k])


def mine_pairs(queries: Sequence[Query], products: Sequence[Product], topk: int = 20, per_query_pairs: int = 24, seed: int = 0):
    rng = random.Random(seed)
    proj = build_token_projection(queries, products, seed=seed)
    pairs = []
    bucket_counts = defaultdict(int)
    for q in queries:
        lex_scores = [lexical_score(q, p) for p in products]
        dense_scores = [dense_baseline_score(q, p, proj) for p in products]
        lex_top = set(topk_indices(lex_scores, k=topk))
        dense_top = set(topk_indices(dense_scores, k=topk))
        union = lex_top | dense_top
        candidates = list(union)
        outside = [i for i in range(len(products)) if i not in union]
        rng.shuffle(outside)
        candidates.extend(outside[: max(8, topk // 2)])
        candidates = list(dict.fromkeys(candidates))
        local = []
        for idx in candidates:
            p = products[idx]
            grade = relevance_grade(q, p)
            in_lex = idx in lex_top
            in_dense = idx in dense_top
            if grade >= 3 and in_lex and in_dense:
                bucket = "easy_pos"
            elif grade >= 3 and in_lex and not in_dense:
                bucket = "hard_pos"
            elif grade == 0 and in_dense and not in_lex:
                bucket = "hard_neg"
            elif grade == 0 and (not in_lex) and (not in_dense):
                bucket = "easy_neg"
            else:
                bucket = "medium"
            local.append({"qid": q.qid, "pid": p.pid, "grade": int(grade), "bucket": bucket})
        by_bucket: Dict[str, List[dict]] = defaultdict(list)
        for item in local:
            by_bucket[item["bucket"]].append(item)
        targets = {
            "easy_pos": per_query_pairs // 6,
            "hard_pos": per_query_pairs // 6,
            "hard_neg": per_query_pairs // 4,
            "medium": per_query_pairs // 4,
            "easy_neg": per_query_pairs // 4,
        }
        picked = []
        for bucket, n in targets.items():
            pool = by_bucket.get(bucket, [])
            rng.shuffle(pool)
            picked.extend(pool[:n])
        if not any(item["grade"] >= 3 for item in picked):
            pos_pool = [item for item in local if item["grade"] >= 3]
            if pos_pool:
                picked.append(rng.choice(pos_pool))
        for item in picked:
            pairs.append(item)
            bucket_counts[item["bucket"]] += 1
    return pairs, dict(bucket_counts)


def _clip_grade(g: int) -> int:
    return int(max(0, min(4, g)))


def noisy_grade(true_grade: int, bucket: str, level: str, rng: random.Random) -> Tuple[int, float]:
    bucket_noise = {
        "easy_pos": 0.10,
        "hard_pos": 0.22,
        "hard_neg": 0.20,
        "medium": 0.16,
        "easy_neg": 0.08,
        "": 0.15,
    }.get(bucket, 0.16)
    level_scale = {"fast": 1.35, "mid": 1.0, "strong": 0.65}.get(level, 1.0)
    sigma = bucket_noise * level_scale
    delta = int(round(rng.gauss(0, sigma * 3.0)))
    pred = _clip_grade(true_grade + delta)
    conf = float(max(0.05, min(0.99, 1.0 - sigma * (abs(delta) + 0.5))))
    return pred, conf


class CalibratedCascade:
    def __init__(self, target_precision: float = 0.85, seed: int = 0):
        self.target_precision = target_precision
        self.rng = random.Random(seed)
        self.thresholds: Dict[int, float] = {g: 0.0 for g in range(5)}

    def fit(self, calibration: Sequence[Tuple[int, str]]) -> None:
        by_pred: Dict[int, List[Tuple[float, bool]]] = defaultdict(list)
        for true_grade, bucket in calibration:
            pred, conf = noisy_grade(true_grade, bucket, "fast", self.rng)
            by_pred[pred].append((conf, pred == true_grade))
        for pred_grade, items in by_pred.items():
            items.sort(key=lambda x: -x[0])
            tp = 0
            for i, (conf, ok) in enumerate(items, start=1):
                tp += 1 if ok else 0
                if tp / i >= self.target_precision:
                    self.thresholds[pred_grade] = conf
            if self.thresholds[pred_grade] == 0.0 and items:
                self.thresholds[pred_grade] = max(0.65, items[0][0] * 0.85)

    def annotate(self, true_grade: int, bucket: str) -> Tuple[int, float]:
        pred, conf = noisy_grade(true_grade, bucket, "fast", self.rng)
        if conf >= self.thresholds.get(pred, 0.7):
            return pred, conf
        pred, conf = noisy_grade(true_grade, bucket, "mid", self.rng)
        if conf >= max(0.55, self.thresholds.get(pred, 0.6) - 0.08):
            return pred, conf
        return noisy_grade(true_grade, bucket, "strong", self.rng)


def run_annotation_cascade(pairs: Sequence[Dict], seed: int = 0) -> List[AnnotatedPair]:
    rng = random.Random(seed)
    cal = [(item["grade"], item.get("bucket", "")) for item in pairs]
    rng.shuffle(cal)
    cal = cal[: max(200, len(cal) // 8)]
    cascade = CalibratedCascade(target_precision=0.85, seed=seed)
    cascade.fit(cal)
    out: List[AnnotatedPair] = []
    for item in pairs:
        pred, conf = cascade.annotate(item["grade"], item.get("bucket", ""))
        out.append(
            AnnotatedPair(
                qid=item["qid"],
                pid=item["pid"],
                true_grade=int(item["grade"]),
                pseudo_grade=int(pred),
                confidence=float(conf),
                bucket=item.get("bucket", ""),
            )
        )
    return out


def build_training_items(annotated_pairs: Sequence[AnnotatedPair], queries_by_id, products_by_id, vocab):
    items = []
    for item in annotated_pairs:
        q = queries_by_id[item.qid]
        p = products_by_id[item.pid]
        items.append(
            {
                "q_ids": encode(q.text, vocab),
                "d_ids": encode(p.title, vocab),
                "label": 1.0 if item.pseudo_grade >= 2 else 0.0,
                "grade": item.pseudo_grade,
                "bucket": item.bucket,
                "qid": item.qid,
                "pid": item.pid,
            }
        )
    return items


def build_click_like_pairs(train_queries: Sequence[Query], products: Sequence[Product], vocab: Dict[str, int], seed: int = 0):
    rng = random.Random(seed)
    items = []
    for q in train_queries:
        positives = [p for p in products if relevance_grade(q, p) >= 2]
        negatives = [p for p in products if relevance_grade(q, p) == 0]
        if not positives or len(negatives) < 3:
            continue
        p = rng.choice(positives)
        items.append({"q_ids": encode(q.text, vocab), "d_ids": encode(p.title, vocab), "label": 1.0, "grade": 4, "bucket": "click_pos"})
        for n in rng.sample(negatives, k=3):
            items.append({"q_ids": encode(q.text, vocab), "d_ids": encode(n.title, vocab), "label": 0.0, "grade": 0, "bucket": "click_neg"})
    return items


def build_mnr_pairs(annotated_pairs: Sequence[AnnotatedPair], queries_by_id, products_by_id, vocab):
    best_pos = {}
    for item in annotated_pairs:
        if item.pseudo_grade < 3:
            continue
        if item.qid not in best_pos or item.confidence > best_pos[item.qid].confidence:
            best_pos[item.qid] = item
    pairs = []
    for item in best_pos.values():
        q = queries_by_id[item.qid]
        p = products_by_id[item.pid]
        pairs.append((encode(q.text, vocab), encode(p.title, vocab)))
    return pairs


def build_triplets(annotated_pairs: Sequence[AnnotatedPair], queries_by_id, products_by_id, vocab):
    by_q = defaultdict(list)
    for item in annotated_pairs:
        by_q[item.qid].append(item)
    triplets = []
    rng = random.Random(0)
    for qid, items in by_q.items():
        pos = [item for item in items if item.pseudo_grade >= 3]
        neg = [item for item in items if item.pseudo_grade <= 1 or item.bucket == "hard_neg"]
        if not pos or not neg:
            continue
        for _ in range(2):
            p_item = rng.choice(pos)
            n_item = rng.choice(neg)
            q = queries_by_id[qid]
            p = products_by_id[p_item.pid]
            n = products_by_id[n_item.pid]
            triplets.append((encode(q.text, vocab), encode(p.title, vocab), encode(n.title, vocab)))
    return triplets


def ndcg_at_k(grades: Sequence[int], k: int = 10) -> float:
    k = min(k, len(grades))
    gains = np.array([(2 ** g - 1) for g in grades[:k]], dtype=np.float32)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float((gains * discounts).sum())
    ideal = sorted(grades, reverse=True)
    ideal_g = np.array([(2 ** g - 1) for g in ideal[:k]], dtype=np.float32)
    idcg = float((ideal_g * discounts).sum())
    return 0.0 if idcg <= 1e-9 else dcg / idcg


def recall_at_k(grades: Sequence[int], k: int = 10, positive_grade: int = 2) -> float:
    k = min(k, len(grades))
    return float(any(g >= positive_grade for g in grades[:k]))


@torch.no_grad()
def evaluate_retrieval(model, vocab: Dict[str, int], queries: Sequence[Query], products: Sequence[Product], device: str = "cpu", k: int = 10):
    from model import make_mask, pad_2d

    model.eval()
    model.to(device)
    q_ids = [encode(q.text, vocab) for q in queries]
    d_ids = [encode(p.title, vocab) for p in products]
    q_pad = pad_2d(q_ids, pad_id=0).to(device)
    d_pad = pad_2d(d_ids, pad_id=0).to(device)
    q_mask = make_mask(q_pad).to(device)
    d_mask = make_mask(d_pad).to(device)
    q_emb = model.encode_query(q_pad, q_mask)
    d_emb = model.encode_doc(d_pad, d_mask)
    sim = (q_emb @ d_emb.T).cpu().numpy()

    ndcgs = []
    recalls = []
    tail_ndcgs = []
    tail_recalls = []
    embarrassing = []

    brand_freq: Dict[str, int] = {}
    for q in queries:
        b = q.intent.get("brand")
        brand_freq[b] = brand_freq.get(b, 0) + 1
    rare_brands = {b for b, c in brand_freq.items() if c <= max(1, int(len(queries) * 0.05))}

    for i, q in enumerate(queries):
        order = np.argsort(-sim[i])
        ranked = [relevance_grade(q, products[j]) for j in order]
        nd = ndcg_at_k(ranked, k=k)
        rc = recall_at_k(ranked, k=k)
        ndcgs.append(nd)
        recalls.append(rc)
        embarrassing.append(float(ranked[0] == 0))
        if q.intent.get("brand") in rare_brands:
            tail_ndcgs.append(nd)
            tail_recalls.append(rc)
    return {
        f"ndcg@{k}": float(np.mean(ndcgs)),
        f"recall@{k}": float(np.mean(recalls)),
        f"tail_ndcg@{k}": float(np.mean(tail_ndcgs) if tail_ndcgs else 0.0),
        f"tail_recall@{k}": float(np.mean(tail_recalls) if tail_recalls else 0.0),
        "embarrassing@1": float(np.mean(embarrassing)),
    }


def build_cfg_from_ckpt(payload: Dict) -> DataConfig:
    return DataConfig(**payload["data_cfg"])
