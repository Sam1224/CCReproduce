import os
import math
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


# -----------------------------
# Utils
# -----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Vocab:
    def __init__(self, tokens: List[str], special_tokens: Optional[List[str]] = None):
        if special_tokens is None:
            special_tokens = ["<pad>", "<unk>"]
        uniq = []
        seen = set()
        for t in special_tokens + tokens:
            if t not in seen:
                seen.add(t)
                uniq.append(t)
        self.itos = uniq
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.pad_id = self.stoi.get("<pad>")
        self.unk_id = self.stoi.get("<unk>")

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> List[int]:
        if self.unk_id is None:
            raise ValueError("This vocab has no <unk> token, encode() is not supported.")
        ids = []
        for w in text.strip().split():
            ids.append(self.stoi.get(w, self.unk_id))
        return ids

    def decode(self, ids: List[int]) -> str:
        return " ".join(self.itos[i] for i in ids)

    def to_dict(self) -> Dict[str, Any]:
        return {"itos": self.itos}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Vocab":
        # keep the same pad/unk ids
        itos = d["itos"]
        v = Vocab(tokens=[])
        v.itos = itos
        v.stoi = {t: i for i, t in enumerate(itos)}
        v.pad_id = v.stoi.get("<pad>")
        v.unk_id = v.stoi.get("<unk>")
        return v


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


# -----------------------------
# Simple numpy k-means (toy)
# -----------------------------

def kmeans_np(x: np.ndarray, k: int, iters: int = 30, seed: int = 0) -> np.ndarray:
    """A tiny k-means implementation (L2, random init). Returns cluster labels."""
    rng = np.random.RandomState(seed)
    n, d = x.shape
    # init centroids
    centroids = x[rng.choice(n, size=k, replace=False)].copy()

    for _ in range(iters):
        # assign
        dist = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)  # (n, k)
        labels = dist.argmin(axis=1)

        # update
        new_centroids = centroids.copy()
        for j in range(k):
            idx = np.where(labels == j)[0]
            if len(idx) == 0:
                new_centroids[j] = x[rng.randint(0, n)]
            else:
                new_centroids[j] = x[idx].mean(axis=0)

        if np.allclose(new_centroids, centroids):
            centroids = new_centroids
            break
        centroids = new_centroids

    # final assignment
    dist = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
    labels = dist.argmin(axis=1)
    return labels.astype(np.int64)


# -----------------------------
# Data generation
# -----------------------------

@dataclass
class ToyConfig:
    seed: int = 42
    n_clusters: int = 24
    docs_per_cluster: int = 20
    cluster_vocab: int = 24
    common_vocab: int = 40
    doc_len_cluster_words: int = 8
    doc_len_common_words: int = 2
    noise_words: int = 1  # pick from other clusters to make clustering non-trivial

    train_queries_per_cluster: int = 60
    test_queries_per_cluster: int = 30

    # 训练中对高 value 文档的偏好强度（越大越偏向高 value）
    value_alpha: float = 3.0

    # 用指数衰减把“簇内 value 排名差距”拉大：value = exp(-rank / value_tau)
    # value_tau 越小，头部越陡峭，CRID 的收益越明显。
    value_tau: float = 3.0

    # 部分簇：训练时不暴露 top-k 高价值文档（模拟"新上架/冷启动"的高价值 item 没被点过）
    holdout_ratio: float = 0.35
    holdout_topk: int = 2


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / (e.sum() + 1e-12)


def generate_toy_data(cfg: ToyConfig) -> Dict[str, Any]:
    """Generate docs + queries; build SID/CRID mapping.

    Returned artifact contains everything needed for both training and testing.
    """
    set_seed(cfg.seed)

    common_words = [f"common{j}" for j in range(cfg.common_vocab)]
    cluster_words: Dict[int, List[str]] = {
        c: [f"topic{c}_w{i}" for i in range(cfg.cluster_vocab)] for c in range(cfg.n_clusters)
    }

    # Create docs
    docs: List[Dict[str, Any]] = []
    all_words = common_words + [w for c in range(cfg.n_clusters) for w in cluster_words[c]]
    word2idx = {w: i for i, w in enumerate(all_words)}

    # cluster-level value prior (some topics are more valuable)
    topic_value_bias = np.random.uniform(0.0, 0.6, size=(cfg.n_clusters,)).astype(np.float32)

    for true_c in range(cfg.n_clusters):
        for d in range(cfg.docs_per_cluster):
            cw = random.sample(cluster_words[true_c], cfg.doc_len_cluster_words)
            mw = random.sample(common_words, cfg.doc_len_common_words)
            nw = []
            if cfg.noise_words > 0:
                other_c = random.choice([x for x in range(cfg.n_clusters) if x != true_c])
                nw = random.sample(cluster_words[other_c], cfg.noise_words)

            text_words = cw + mw + nw
            random.shuffle(text_words)

            # business value in [0, 1]
            base = np.float32(np.random.beta(2.0, 5.0))  # skew to low values
            value = float(np.clip(base + topic_value_bias[true_c], 0.0, 1.0))

            docs.append(
                {
                    "doc_idx": len(docs),
                    "true_topic": true_c,
                    "text": " ".join(text_words),
                    "value": value,
                }
            )

    # doc embeddings = bag-of-words counts (toy)
    x = np.zeros((len(docs), len(all_words)), dtype=np.float32)
    for i, doc in enumerate(docs):
        for w in doc["text"].split():
            x[i, word2idx[w]] += 1.0
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

    # semantic clustering => prefix
    semantic_labels = kmeans_np(x, k=cfg.n_clusters, iters=40, seed=cfg.seed)
    for i, doc in enumerate(docs):
        doc["cluster"] = int(semantic_labels[i])

    # keywords per semantic cluster (used for query generation)
    cluster_keywords: Dict[int, List[str]] = {}
    for c in range(cfg.n_clusters):
        idx = [i for i, d in enumerate(docs) if d["cluster"] == c]
        if len(idx) == 0:
            cluster_keywords[c] = random.sample(all_words, 6)
            continue

        freq = x[idx].sum(axis=0)
        # prefer topic words, avoid too many common words
        cand = []
        for wid in np.argsort(-freq):
            w = all_words[int(wid)]
            if w.startswith("topic"):
                cand.append(w)
            if len(cand) >= 12:
                break
        if len(cand) < 6:
            cand += random.sample(common_words, 6 - len(cand))
        cluster_keywords[c] = cand

    # within-cluster ranking by business value => CRID suffix
    cluster_to_docs: Dict[int, List[int]] = {c: [] for c in range(cfg.n_clusters)}
    for d in docs:
        cluster_to_docs[d["cluster"]].append(d["doc_idx"])

    # For each semantic cluster: sort docs by value desc; assign rank 0..n-1
    cluster_ranked_docs: Dict[int, List[int]] = {}
    for c, idxs in cluster_to_docs.items():
        idxs_sorted = sorted(idxs, key=lambda i: docs[i]["value"], reverse=True)
        cluster_ranked_docs[c] = idxs_sorted
        for r, doc_idx in enumerate(idxs_sorted):
            docs[doc_idx]["rank"] = int(r)

    # 让业务价值与 rank 呈指数衰减，制造明显的“头部价值”效应，便于观察 CRID 的排序收益。
    # 注意：我们用上面得到的 rank 来重设 value（保持排序一致）。
    for c, idxs_sorted in cluster_ranked_docs.items():
        for r, doc_idx in enumerate(idxs_sorted):
            docs[doc_idx]["value"] = float(math.exp(-float(r) / float(cfg.value_tau)))

    # SID: same semantic prefix, but suffix is a random permutation of ranks (no value ordering)
    sid_perm: Dict[int, List[int]] = {}
    for c in range(cfg.n_clusters):
        m = len(cluster_ranked_docs[c])
        perm = list(range(m))
        random.shuffle(perm)
        sid_perm[c] = perm  # rank -> sid_suffix_idx

    # Build doc IDs and lookup
    max_suffix = max(len(v) for v in cluster_ranked_docs.values())

    lookup_crid: Dict[Tuple[int, int], int] = {}  # (cluster, rank) -> doc_idx
    lookup_sid: Dict[Tuple[int, int], int] = {}  # (cluster, sid_suffix_idx) -> doc_idx

    for doc in docs:
        c = doc["cluster"]
        r = doc["rank"]
        sid_suffix = sid_perm[c][r]
        doc["crid"] = f"C{c:02d}-R{r:02d}"
        doc["sid"] = f"C{c:02d}-S{sid_suffix:02d}"
        doc["sid_suffix"] = int(sid_suffix)

        lookup_crid[(c, r)] = doc["doc_idx"]
        lookup_sid[(c, sid_suffix)] = doc["doc_idx"]

    # choose holdout clusters where top-value docs are never used as training targets
    n_holdout = int(round(cfg.n_clusters * cfg.holdout_ratio))
    holdout_clusters = set(random.sample(list(range(cfg.n_clusters)), k=max(1, n_holdout)))

    def sample_train_target(cluster_id: int) -> int:
        ranked = cluster_ranked_docs[cluster_id]
        values = np.array([docs[i]["value"] for i in ranked], dtype=np.float32)

        # optionally hold out top-k docs
        start = cfg.holdout_topk if cluster_id in holdout_clusters else 0
        cand = ranked[start:]
        cand_values = values[start:]
        if len(cand) == 0:
            cand = ranked
            cand_values = values

        # 用幂函数把高 value 的权重拉开（比 softmax(alpha*value) 更不容易被“大簇尾部数量”淹没）
        weights = np.power(cand_values, cfg.value_alpha)
        probs = weights / (weights.sum() + 1e-12)
        return int(np.random.choice(cand, p=probs))

    templates_train = [
        "请检索 {w1} {w2} 相关 的 文档",
        "我想了解 {w1} 和 {w2} 的资料",
        "有没有关于 {w1} {w2} 的内容",
        "寻找 {w1} {w2} 的说明",
    ]
    templates_test = [
        "帮我找一下 {w1} {w2} 主题 的 文档",
        "查询 {w1} 与 {w2} 的相关信息",
        "需要 {w1} {w2} 的介绍",
    ]

    train_queries: List[Dict[str, Any]] = []
    test_queries: List[Dict[str, Any]] = []

    for c in range(cfg.n_clusters):
        kws = cluster_keywords[c]

        for _ in range(cfg.train_queries_per_cluster):
            w1, w2 = random.sample(kws, 2)
            q = random.choice(templates_train).format(w1=w1, w2=w2)
            target = sample_train_target(c)
            train_queries.append({"query": q, "cluster": c, "target_doc": target})

        for _ in range(cfg.test_queries_per_cluster):
            w1, w2 = random.sample(kws, 2)
            q = random.choice(templates_test).format(w1=w1, w2=w2)
            # For testing we don't need a single target; keep one for debugging
            target = int(cluster_ranked_docs[c][0])  # top value
            test_queries.append({"query": q, "cluster": c, "target_doc": target})

    # Build query vocab from all queries
    all_q_words = []
    for item in train_queries + test_queries:
        all_q_words.extend(item["query"].split())
    q_vocab_tokens = sorted(set(all_q_words))
    query_vocab = Vocab(tokens=q_vocab_tokens)

    # ID vocabs are global tokens: prefix Cxx; suffix Rxx or Sxx (0..max_suffix-1)
    prefix_tokens = [f"C{c:02d}" for c in range(cfg.n_clusters)]
    crid_suffix_tokens = [f"R{s:02d}" for s in range(max_suffix)]
    sid_suffix_tokens = [f"S{s:02d}" for s in range(max_suffix)]

    # DocID token vocab 不需要 <pad>/<unk>，否则 beam search 会浪费候选并引入无效 token。
    prefix_vocab = Vocab(tokens=prefix_tokens, special_tokens=[])
    crid_suffix_vocab = Vocab(tokens=crid_suffix_tokens, special_tokens=[])
    sid_suffix_vocab = Vocab(tokens=sid_suffix_tokens, special_tokens=[])

    artifact: Dict[str, Any] = {
        "cfg": cfg.__dict__,
        "docs": docs,
        "cluster_ranked_docs": cluster_ranked_docs,
        "holdout_clusters": sorted(list(holdout_clusters)),
        "lookup": {
            "crid": {f"{c}-{r}": di for (c, r), di in lookup_crid.items()},
            "sid": {f"{c}-{s}": di for (c, s), di in lookup_sid.items()},
        },
        "queries": {"train": train_queries, "test": test_queries},
        "vocabs": {
            "query": query_vocab.to_dict(),
            "prefix": prefix_vocab.to_dict(),
            "crid_suffix": crid_suffix_vocab.to_dict(),
            "sid_suffix": sid_suffix_vocab.to_dict(),
        },
    }
    return artifact


def save_artifact(artifact: Dict[str, Any], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    torch.save(artifact, path)


def load_artifact(path: str) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu")


# -----------------------------
# Torch dataset
# -----------------------------

class GenerativeRetrievalDataset(Dataset):
    def __init__(
        self,
        queries: List[Dict[str, Any]],
        docs: List[Dict[str, Any]],
        query_vocab: Vocab,
        prefix_vocab: Vocab,
        suffix_vocab: Vocab,
        id_scheme: str,
    ):
        assert id_scheme in ["crid", "sid"]
        self.queries = queries
        self.docs = docs
        self.query_vocab = query_vocab
        self.prefix_vocab = prefix_vocab
        self.suffix_vocab = suffix_vocab
        self.id_scheme = id_scheme

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.queries[idx]
        q_ids = self.query_vocab.encode(item["query"])

        cluster = int(item["cluster"])
        prefix_tok = f"C{cluster:02d}"
        prefix_id = self.prefix_vocab.stoi[prefix_tok]

        # label suffix depends on target doc and id scheme
        doc = self.docs[int(item["target_doc"])]
        if self.id_scheme == "crid":
            suffix_tok = f"R{int(doc['rank']):02d}"
        else:
            suffix_tok = f"S{int(doc['sid_suffix']):02d}"
        suffix_id = self.suffix_vocab.stoi[suffix_tok]

        return {
            "query_ids": torch.tensor(q_ids, dtype=torch.long),
            "length": len(q_ids),
            "prefix_id": torch.tensor(prefix_id, dtype=torch.long),
            "suffix_id": torch.tensor(suffix_id, dtype=torch.long),
            "cluster": cluster,
            "target_doc": int(item["target_doc"]),
        }


def collate_fn(batch: List[Dict[str, Any]], pad_id: int) -> Dict[str, Any]:
    max_len = max(x["length"] for x in batch)
    q = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    lengths = torch.tensor([x["length"] for x in batch], dtype=torch.long)
    prefix = torch.stack([x["prefix_id"] for x in batch])
    suffix = torch.stack([x["suffix_id"] for x in batch])

    for i, x in enumerate(batch):
        q[i, : x["length"]] = x["query_ids"]

    return {
        "query_ids": q,
        "lengths": lengths,
        "prefix_id": prefix,
        "suffix_id": suffix,
    }


def build_vocabs_from_artifact(artifact: Dict[str, Any], id_scheme: str) -> Tuple[Vocab, Vocab, Vocab]:
    vocabs = artifact["vocabs"]
    query_vocab = Vocab.from_dict(vocabs["query"])
    prefix_vocab = Vocab.from_dict(vocabs["prefix"])
    if id_scheme == "crid":
        suffix_vocab = Vocab.from_dict(vocabs["crid_suffix"])
    else:
        suffix_vocab = Vocab.from_dict(vocabs["sid_suffix"])
    return query_vocab, prefix_vocab, suffix_vocab
