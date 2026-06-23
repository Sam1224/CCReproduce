"""
data.py — Toy dataset + synthetic data generator for OneBar reproduction.

This module faithfully mirrors OneBar's *Collaborative-Multimodal Intent
Grounding* (paper Sec. 3.2) and the *evidence-schema serialization /
prompt-compression* of Sec. 3.3.

Each video impression is abstracted into a structured evidence schema:

    E(x, u) = < T_x , M_x , A_x , H_u >

      T_x : cleaned textual title  (strip @mentions / #hashtags / promo tokens)
      M_x : multimodal video summary (objects / scene / ASR / creator)
      A_x : collaborative query anchors (behavior-aligned ANN anchors:
            q->v, v->q, q->q relations ; see Table 1 of the paper)
      H_u : relevance-filtered user history queries

and serialized (Eq. 4) into a compact, [SEP]-delimited prompt:

    s_x = [ T_x ; [SEP] ; M_x ; [SEP] ; A_x ; [SEP] ; H_u ]

This *field-aligned compressed* prompt is a key ablation in the paper:
HR@8 0.3564 (compressed) vs. 0.1864 (verbose instruction prompt) — see
`build_prompt(..., style="compressed"|"verbose")`.

For each trigger we also build a behavior-induced ordinal preference list
over the 6 hierarchical behavior levels (paper Table 2):

    1  clicked AND post-click engaged by the CURRENT user   (strongest)
    2  clicked by the current user
    3  clicked AND post-click engaged by OTHER users
    4  clicked by other users
    5  exposed but not clicked
    6  retrieved but unexposed                                (weakest)

A *monotonic CTR filter* (Sec. 3.4.2) drops candidates whose historical CTR
contradicts the behavior order, and keeps one representative per level.

NOTE on faithfulness / approximations:
  * The real system summarizes sampled frames + ASR with a multimodal
    foundation model (M_x) and trains a Qwen3-Embedding 0.6B with staged
    behavior supervision to build the 128-d ANN space (A_x). Reproducing
    those production components is out of scope for a CPU smoke test, so we
    synthesize plausible M_x / A_x from category templates. The *interface*
    (fields, serialization, labels) is identical to the paper so the model
    and training code are unchanged. See `stub_multimodal_summarizer` and
    `stub_behavior_aligned_ann` for the real-pipeline pseudocode.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional

import torch
from torch.utils.data import Dataset

SEP = " [SEP] "
REJECT_TOKEN = "[REJECT]"

# Behavior level -> base CTR (monotone non-increasing) used to synthesize logs.
LEVEL_BASE_CTR = {1: 0.42, 2: 0.30, 3: 0.20, 4: 0.12, 5: 0.03, 6: 0.0}


# --------------------------------------------------------------------------- #
# Category templates used by the synthetic generator.
# Each entry mimics one e-commerce short-video vertical.
# --------------------------------------------------------------------------- #
CATEGORIES: Dict[str, Dict] = {
    "makeup": {
        "titles": [
            "@beautyGuru 教你画 #秋冬妆容 哑光唇釉测评 福利领取",
            "新手必看！持妆12小时底妆教程 #美妆 @官方旗舰店 限时5折",
        ],
        "objects": ["lipstick", "foundation", "makeup brush", "mirror"],
        "scene": ["bedroom vanity", "soft ring light"],
        "asr": ["matte texture", "long lasting", "dry lips friendly"],
        "creator": "beauty creator",
        "queries": [
            "哑光唇釉 不拔干", "持久不脱妆粉底液", "新手底妆教程",
            "秋冬显白口红", "控油散粉", "美妆蛋推荐",
        ],
    },
    "coffee": {
        "titles": [
            "@coffeeLife #拉花教程 在家也能做拿铁 家用咖啡机种草 双十一优惠",
            "手冲咖啡入门 #咖啡 @某品牌 三秒出杯 点击领券",
        ],
        "objects": ["espresso machine", "milk pitcher", "latte art", "beans"],
        "scene": ["warm tone cafe", "morning sunlight"],
        "asr": ["latte art", "home barista", "fresh beans"],
        "creator": "coffee blogger",
        "queries": [
            "家用咖啡机推荐", "拉花奶缸", "手冲咖啡器具",
            "意式浓缩咖啡豆", "半自动咖啡机", "咖啡入门套装",
        ],
    },
    "sneaker": {
        "titles": [
            "@sneakerHead #球鞋开箱 复古跑鞋上脚 实测 评论区抽奖",
            "夏季透气跑鞋测评 #运动 @旗舰店 满300减50",
        ],
        "objects": ["running shoes", "shoe box", "outsole"],
        "scene": ["outdoor track", "city street"],
        "asr": ["breathable", "cushioning", "lightweight"],
        "creator": "sneaker reviewer",
        "queries": [
            "透气跑鞋推荐", "复古老爹鞋", "缓震跑步鞋",
            "夏季运动鞋", "百搭小白鞋", "专业马拉松鞋",
        ],
    },
    "kitchen": {
        "titles": [
            "@homeCook #厨房好物 空气炸锅食谱 懒人快手菜 关注不迷路",
            "多功能料理锅测评 #美食 @品牌方 买一送一",
        ],
        "objects": ["air fryer", "cooking pot", "vegetables"],
        "scene": ["modern kitchen", "bright counter"],
        "asr": ["quick recipe", "oil free", "easy clean"],
        "creator": "home cook",
        "queries": [
            "空气炸锅推荐", "多功能料理锅", "懒人厨房好物",
            "不粘锅推荐", "空气炸锅食谱", "电煮锅小型",
        ],
    },
    # A "policy-sensitive" category that should map to [REJECT] (Sec. 3.4.1).
    "_reject": {
        "titles": [
            "@randomUser 震惊！点击查看 #违规内容 私聊领取 福利群",
            "高仿代购 #敏感 @不明账号 加微信 暴利项目",
        ],
        "objects": ["unknown"],
        "scene": ["unclear"],
        "asr": ["spammy promo"],
        "creator": "spam account",
        "queries": [],  # no safe query -> [REJECT]
    },
}

# A global "historical query corpus" used to draw ANN anchors / cross-category
# noise — mimics the production query pool the ANN index is built over.
GLOBAL_QUERY_POOL: List[str] = sum([c["queries"] for c in CATEGORIES.values()], [])


# --------------------------------------------------------------------------- #
# Text cleaning (T_x) — Sec. 3.2 "Textual and multimodal evidence".
# --------------------------------------------------------------------------- #
_MENTION = re.compile(r"@\S+")
_HASHTAG = re.compile(r"#\S+")
_PROMO = re.compile(
    r"(福利|领取|限时|折|满\d+减\d+|点击|关注不迷路|抽奖|优惠|买一送一|"
    r"双十一|领券|私聊|加微信|暴利项目|种草|福利群)"
)


def clean_title(raw: str) -> str:
    """Form T_x: drop @mentions, #hashtags, promo/boilerplate tokens, collapse ws."""
    t = _MENTION.sub(" ", raw)
    t = _HASHTAG.sub(" ", t)
    t = _PROMO.sub(" ", t)
    t = re.sub(r"[！!。，,、]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# --------------------------------------------------------------------------- #
# Production-component stubs (clearly documented, NOT silently broken).
# --------------------------------------------------------------------------- #
def stub_multimodal_summarizer(frames, asr) -> str:
    """STUB — real pipeline (paper Sec. 3.2 / Fig. 3 "Multimodal Info Fusion").

    Real implementation (pseudocode):
        frames = sample_keyframes(video, n=8)
        visual = MLLM.describe(frames)            # objects, OCR, scene, product usage
        speech = ASR(audio)                       # narration / BGM cues
        M_x = MLLM.summarize(prompt(visual, speech, "extract search intent"))
        M_x = strip_template_prefix(M_x)          # refreshed daily, incremental
    Here we deterministically compose M_x from category templates instead.
    """
    raise NotImplementedError("Replaced by template synthesis in the toy generator.")


def stub_behavior_aligned_ann(trigger_emb, query_index, sim_thresh=0.88):
    """STUB — behavior-aligned ANN anchor retrieval (paper Sec. 3.2, Table 1).

    Real implementation (pseudocode):
        # 1. train Qwen3-Embedding-0.6B with staged behavior supervision:
        #      stage a: q->v  (search-click logs)  -> relevance-preserving space
        #      stage b: += v->q (post-video search) and q->q (reformulation)
        #    project into a shared 128-d space.
        # 2. offline: for each trigger, retrieve queries of its nearest video
        #    triggers; keep anchors with trigger similarity > 0.88
        #    (raises anchor hit-rate 0.282 -> 0.335).
        nbrs = ann_index.search(trigger_emb, topk)
        anchors = [q for (q, sim) in nbrs if sim > sim_thresh]
    Here we sample plausible anchors from the same category's query set.
    """
    raise NotImplementedError("Replaced by template synthesis in the toy generator.")


# --------------------------------------------------------------------------- #
# Synthetic sample
# --------------------------------------------------------------------------- #
@dataclass
class OneBarSample:
    trigger_id: str
    user_id: str
    category: str
    raw_title: str
    T_x: str                      # cleaned title
    M_x: str                      # multimodal summary
    A_x: List[str]                # collaborative anchors
    H_u: List[str]                # relevance-filtered user history
    clicked_query: str            # y+  (or [REJECT])
    is_reject: bool
    # preference list: list of dicts {query, level, ctr}
    preference_list: List[Dict] = field(default_factory=list)

    def to_json(self) -> Dict:
        return asdict(self)


def _mk_summary(cat_cfg: Dict) -> str:
    objs = ", ".join(cat_cfg["objects"])
    scene = ", ".join(cat_cfg["scene"])
    asr = ", ".join(cat_cfg["asr"])
    return f"objects: {objs}; scene: {scene}; asr: {asr}; creator: {cat_cfg['creator']}"


def _build_preference_list(rng: random.Random, cat_cfg: Dict) -> List[Dict]:
    """Build a 6-level behavior list and apply monotonic-CTR filtering (Sec. 3.4.2)."""
    qs = list(cat_cfg["queries"])
    rng.shuffle(qs)
    # cross-category distractors as low-level (retrieved-but-unexposed) candidates.
    distractors = rng.sample([q for q in GLOBAL_QUERY_POOL if q not in qs], k=2)

    raw = []
    # assign top relevant queries to higher levels
    level_queries = {
        1: qs[0:1],
        2: qs[1:2],
        3: qs[2:3],
        4: qs[3:4],
        5: qs[4:5],
        6: distractors[:1],
    }
    for lvl, items in level_queries.items():
        for q in items:
            # noisy CTR around the level base value
            base = LEVEL_BASE_CTR[lvl]
            ctr = max(0.0, base + rng.uniform(-0.05, 0.05))
            raw.append({"query": q, "level": lvl, "ctr": round(ctr, 4)})

    return monotonic_ctr_filter(raw)


def monotonic_ctr_filter(cands: List[Dict]) -> List[Dict]:
    """Enforce CTR monotonicity wrt behavior level + keep 1 representative/level.

    Paper Sec. 3.4.2: drop candidates whose CTR contradicts the monotone order
    implied by their behavior level; after de-dup keep the highest-CTR
    representative of each level. Returns a list sorted by level (best first).
    """
    # keep best-CTR per level
    by_level: Dict[int, Dict] = {}
    for c in cands:
        lvl = c["level"]
        if lvl not in by_level or c["ctr"] > by_level[lvl]["ctr"]:
            by_level[lvl] = c
    ordered = [by_level[l] for l in sorted(by_level)]

    # monotonic filter: CTR must be non-increasing as level worsens.
    filtered: List[Dict] = []
    prev_ctr = float("inf")
    for c in ordered:
        if c["ctr"] <= prev_ctr + 1e-9:
            filtered.append(c)
            prev_ctr = c["ctr"]
        # else: contradicts behavior order -> dropped
    return filtered


def generate_dataset(n: int = 64, seed: int = 0,
                     reject_ratio: float = 0.12) -> List[OneBarSample]:
    """Synthesize `n` OneBar impressions across categories."""
    rng = random.Random(seed)
    real_cats = [c for c in CATEGORIES if not c.startswith("_")]
    samples: List[OneBarSample] = []
    for i in range(n):
        is_reject = rng.random() < reject_ratio
        cat = "_reject" if is_reject else rng.choice(real_cats)
        cfg = CATEGORIES[cat]
        raw_title = rng.choice(cfg["titles"])
        T_x = clean_title(raw_title)
        M_x = _mk_summary(cfg)

        if is_reject:
            A_x, H_u, pref = [], [], []
            clicked = REJECT_TOKEN
        else:
            pref = _build_preference_list(rng, cfg)
            # collaborative anchors: in-category queries + 1 cross-category noise
            anchors = rng.sample(cfg["queries"], k=min(3, len(cfg["queries"])))
            anchors.append(rng.choice(GLOBAL_QUERY_POOL))
            A_x = anchors
            # relevance-filtered user history (keep mostly in-category, Sec.3.2 H_u)
            H_u = rng.sample(cfg["queries"], k=min(2, len(cfg["queries"])))
            # y+ : level-1 (or best available) clicked query
            clicked = pref[0]["query"] if pref else rng.choice(cfg["queries"])

        samples.append(OneBarSample(
            trigger_id=f"photo_{i:05d}",
            user_id=f"user_{rng.randint(0, 999):03d}",
            category=cat,
            raw_title=raw_title,
            T_x=T_x, M_x=M_x, A_x=A_x, H_u=H_u,
            clicked_query=clicked, is_reject=is_reject,
            preference_list=pref,
        ))
    return samples


# --------------------------------------------------------------------------- #
# Prompt serialization / compression (Eq. 4)
# --------------------------------------------------------------------------- #
def build_prompt(s: OneBarSample, style: str = "compressed") -> str:
    """Serialize evidence schema E into the encoder input s_x.

    style="compressed": field-aligned [SEP]-delimited schema (Eq. 4) — the
        production prompt. Sparse fields are simply omitted.
    style="verbose":   instruction-style natural-language prompt — the weak
        ablation baseline (HR@8 0.1864 vs 0.3564 in the paper, Table 4).
    """
    A_x = ", ".join(s.A_x)
    H_u = ", ".join(s.H_u)
    if style == "compressed":
        parts = [p for p in [s.T_x, s.M_x, A_x, H_u] if p]
        return SEP.join(parts)
    elif style == "verbose":
        return (
            "You are an e-commerce query recommendation assistant. Given the "
            "following short video information, please generate the most likely "
            "search query that the user would click on the bottom bar.\n"
            f"The cleaned title of the video is: {s.T_x}.\n"
            f"A multimodal summary of the video content is: {s.M_x}.\n"
            f"Some collaboratively retrieved candidate queries are: {A_x}.\n"
            f"The user's relevant search history is: {H_u}.\n"
            "Now, generate the recommended search query:"
        )
    raise ValueError(f"unknown prompt style: {style}")


# --------------------------------------------------------------------------- #
# Torch Datasets
# --------------------------------------------------------------------------- #
class SFTDataset(Dataset):
    """Stage-1 SFT: (s_x -> y+) and (s_x -> [REJECT]) pairs."""

    def __init__(self, samples: List[OneBarSample], tokenizer,
                 max_src=128, max_tgt=24, prompt_style="compressed"):
        self.samples = samples
        self.tok = tokenizer
        self.max_src = max_src
        self.max_tgt = max_tgt
        self.prompt_style = prompt_style

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        src = build_prompt(s, self.prompt_style)
        tgt = s.clicked_query
        enc = self.tok(src, max_length=self.max_src, truncation=True,
                       padding="max_length", return_tensors="pt")
        with self.tok.as_target_tokenizer() if hasattr(
                self.tok, "as_target_tokenizer") else _nullctx():
            lab = self.tok(tgt, max_length=self.max_tgt, truncation=True,
                           padding="max_length", return_tensors="pt")
        labels = lab["input_ids"].squeeze(0)
        labels[labels == self.tok.pad_token_id] = -100  # ignore pad in CE
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": labels,
            "target_text": tgt,
        }


class ListwiseDataset(Dataset):
    """Stage-2: per-trigger preference list (queries + behavior levels)."""

    def __init__(self, samples: List[OneBarSample], tokenizer,
                 max_src=128, max_tgt=24, prompt_style="compressed"):
        # only triggers that have >=2 candidates form a usable list
        self.samples = [s for s in samples
                        if not s.is_reject and len(s.preference_list) >= 2]
        self.tok = tokenizer
        self.max_src = max_src
        self.max_tgt = max_tgt
        self.prompt_style = prompt_style

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        src = build_prompt(s, self.prompt_style)
        enc = self.tok(src, max_length=self.max_src, truncation=True,
                       padding="max_length", return_tensors="pt")
        queries = [c["query"] for c in s.preference_list]
        levels = [c["level"] for c in s.preference_list]
        lab = self.tok(queries, max_length=self.max_tgt, truncation=True,
                       padding="max_length", return_tensors="pt")
        labels = lab["input_ids"].clone()
        labels[labels == self.tok.pad_token_id] = -100
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "cand_input_ids": lab["input_ids"],      # [n_cand, max_tgt]
            "cand_labels": labels,                   # [n_cand, max_tgt]
            "levels": torch.tensor(levels, dtype=torch.long),
        }


class PIOPDDataset(Dataset):
    """Stage-3: (standard context x, clicked target y_ref)."""

    def __init__(self, samples: List[OneBarSample], tokenizer,
                 max_src=128, max_tgt=24, prompt_style="compressed"):
        self.samples = [s for s in samples if not s.is_reject]
        self.tok = tokenizer
        self.max_src = max_src
        self.max_tgt = max_tgt
        self.prompt_style = prompt_style

    def __len__(self):
        return len(self.samples)

    def _teacher_prompt(self, s: OneBarSample) -> str:
        """x^(T) = x ⊕_rand y_ref : insert clicked query at a random field slot.

        Randomized Context Augmentation (Eq. 11) — prevents positional
        shortcuts / surface-form copying by the teacher.
        """
        base_fields = [s.T_x, s.M_x, ", ".join(s.A_x), ", ".join(s.H_u)]
        base_fields = [f for f in base_fields if f]
        pos = random.randint(0, len(base_fields))
        post = f"intent: {s.clicked_query}"
        fields = base_fields[:pos] + [post] + base_fields[pos:]
        return SEP.join(fields)

    def __getitem__(self, idx):
        s = self.samples[idx]
        student_src = build_prompt(s, self.prompt_style)   # x^(S)
        teacher_src = self._teacher_prompt(s)              # x^(T)
        s_enc = self.tok(student_src, max_length=self.max_src, truncation=True,
                         padding="max_length", return_tensors="pt")
        t_enc = self.tok(teacher_src, max_length=self.max_src, truncation=True,
                         padding="max_length", return_tensors="pt")
        lab = self.tok(s.clicked_query, max_length=self.max_tgt, truncation=True,
                       padding="max_length", return_tensors="pt")
        labels = lab["input_ids"].squeeze(0)
        labels[labels == self.tok.pad_token_id] = -100
        return {
            "student_input_ids": s_enc["input_ids"].squeeze(0),
            "student_attention_mask": s_enc["attention_mask"].squeeze(0),
            "teacher_input_ids": t_enc["input_ids"].squeeze(0),
            "teacher_attention_mask": t_enc["attention_mask"].squeeze(0),
            "labels": labels,
            "target_text": s.clicked_query,
        }


class _nullctx:
    def __enter__(self): return None
    def __exit__(self, *a): return False


def listwise_collate(batch):
    """Collate variable-length candidate lists by keeping per-sample tensors."""
    return batch  # handled sample-by-sample in train_listwise.py


# --------------------------------------------------------------------------- #
# Persisting a toy dataset to JSONL
# --------------------------------------------------------------------------- #
def dump_jsonl(samples: List[OneBarSample], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s.to_json(), ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> List[OneBarSample]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            out.append(OneBarSample(**d))
    return out


def get_splits(n_train=80, n_eval=24, seed=0):
    train = generate_dataset(n=n_train, seed=seed)
    eval_ = generate_dataset(n=n_eval, seed=seed + 777)
    return train, eval_


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="toy_data.jsonl")
    args = ap.parse_args()
    samples = generate_dataset(n=args.n, seed=args.seed)
    dump_jsonl(samples, args.out)
    print(f"Wrote {len(samples)} samples -> {args.out}")
    s = samples[0]
    print("\n--- example (compressed prompt) ---")
    print(build_prompt(s, "compressed"))
    print("clicked_query:", s.clicked_query)
    print("preference_list:", s.preference_list)
