from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List


@dataclass
class RetrievalRecord:
    query: str
    item_id: str
    item_title: str
    channels: List[str]
    channel_ranks: Dict[str, int]
    lexical_overlap: float
    token_similarity: float
    teacher_grade: int = -1
    cascade_trace: List[str] = field(default_factory=list)
    bucket: str = ""


def build_raw_records() -> List[RetrievalRecord]:
    return [
        RetrievalRecord("iphone charger fast charging", "a1", "iPhone 15 fast USB-C charger cable", ["lexical", "bm25", "ann"], {"lexical": 1, "bm25": 2, "ann": 3}, 0.92, 0.88),
        RetrievalRecord("iphone charger fast charging", "a2", "MagSafe wireless charger stand", ["lexical", "bm25"], {"lexical": 5, "bm25": 4}, 0.73, 0.71),
        RetrievalRecord("iphone charger fast charging", "a3", "Phone case with camera cover", ["ann"], {"ann": 9}, 0.14, 0.21),
        RetrievalRecord("iphone charger fast charging", "a4", "USB-C wall adapter 30W for iPhone", ["bm25", "ann"], {"bm25": 3, "ann": 4}, 0.79, 0.83),
        RetrievalRecord("iphone charger fast charging", "a5", "Lightning cable organizer pouch", ["lexical"], {"lexical": 8}, 0.43, 0.49),

        RetrievalRecord("running shoes women cushion", "b1", "Women's cushioned running shoes breathable", ["lexical", "bm25", "ann"], {"lexical": 2, "bm25": 1, "ann": 2}, 0.90, 0.91),
        RetrievalRecord("running shoes women cushion", "b2", "Trail hiking boots for women waterproof", ["ann"], {"ann": 7}, 0.32, 0.45),
        RetrievalRecord("running shoes women cushion", "b3", "Women's road running sneakers with foam midsole", ["bm25", "ann"], {"bm25": 3, "ann": 3}, 0.84, 0.87),
        RetrievalRecord("running shoes women cushion", "b4", "Women's yoga leggings high waist", ["ann"], {"ann": 10}, 0.06, 0.19),
        RetrievalRecord("running shoes women cushion", "b5", "Women's athletic walking shoes memory foam", ["lexical"], {"lexical": 6}, 0.61, 0.66),

        RetrievalRecord("protein powder chocolate whey", "c1", "Chocolate whey protein powder 2lb", ["lexical", "bm25", "ann"], {"lexical": 1, "bm25": 1, "ann": 1}, 0.94, 0.93),
        RetrievalRecord("protein powder chocolate whey", "c2", "Vanilla whey isolate protein shake mix", ["ann"], {"ann": 6}, 0.48, 0.58),
        RetrievalRecord("protein powder chocolate whey", "c3", "Chocolate pre-workout energy drink", ["lexical"], {"lexical": 7}, 0.46, 0.52),
        RetrievalRecord("protein powder chocolate whey", "c4", "Plant-based chocolate protein powder", ["bm25", "ann"], {"bm25": 4, "ann": 5}, 0.77, 0.79),
        RetrievalRecord("protein powder chocolate whey", "c5", "Glass shaker bottle for gym drinks", ["ann"], {"ann": 9}, 0.09, 0.18),

        RetrievalRecord("air fryer liners parchment", "d1", "Air fryer parchment paper liners 100 pcs", ["lexical", "bm25", "ann"], {"lexical": 1, "bm25": 2, "ann": 1}, 0.96, 0.95),
        RetrievalRecord("air fryer liners parchment", "d2", "Silicone air fryer basket liner reusable", ["bm25", "ann"], {"bm25": 3, "ann": 3}, 0.71, 0.77),
        RetrievalRecord("air fryer liners parchment", "d3", "Kitchen baking tray non-stick", ["ann"], {"ann": 8}, 0.24, 0.36),
        RetrievalRecord("air fryer liners parchment", "d4", "Parchment baking paper roll for oven", ["lexical"], {"lexical": 5}, 0.57, 0.62),
        RetrievalRecord("air fryer liners parchment", "d5", "Air fryer recipe cookbook", ["ann"], {"ann": 10}, 0.12, 0.16),

        RetrievalRecord("wireless earbuds noise cancelling", "e1", "Wireless earbuds with ANC and charging case", ["lexical", "bm25", "ann"], {"lexical": 1, "bm25": 1, "ann": 2}, 0.91, 0.92),
        RetrievalRecord("wireless earbuds noise cancelling", "e2", "Bluetooth earphones sports fit sweatproof", ["ann"], {"ann": 7}, 0.43, 0.59),
        RetrievalRecord("wireless earbuds noise cancelling", "e3", "Noise cancelling over-ear headphones", ["lexical"], {"lexical": 4}, 0.63, 0.69),
        RetrievalRecord("wireless earbuds noise cancelling", "e4", "Wired gaming headset with microphone", ["ann"], {"ann": 9}, 0.11, 0.14),
        RetrievalRecord("wireless earbuds noise cancelling", "e5", "True wireless earbuds hybrid ANC premium", ["bm25", "ann"], {"bm25": 3, "ann": 4}, 0.86, 0.88),

        RetrievalRecord("cat food grain free salmon", "f1", "Grain free salmon cat food dry formula", ["lexical", "bm25", "ann"], {"lexical": 2, "bm25": 1, "ann": 2}, 0.93, 0.90),
        RetrievalRecord("cat food grain free salmon", "f2", "Grain free tuna wet cat food variety pack", ["ann"], {"ann": 6}, 0.56, 0.67),
        RetrievalRecord("cat food grain free salmon", "f3", "Salmon dog treats training bites", ["lexical"], {"lexical": 7}, 0.51, 0.57),
        RetrievalRecord("cat food grain free salmon", "f4", "Limited ingredient salmon cat kibble", ["bm25", "ann"], {"bm25": 3, "ann": 4}, 0.85, 0.86),
        RetrievalRecord("cat food grain free salmon", "f5", "Automatic pet feeder with timer", ["ann"], {"ann": 10}, 0.08, 0.12),
    ]


def group_by_query(records: Iterable[RetrievalRecord]) -> Dict[str, List[RetrievalRecord]]:
    grouped: Dict[str, List[RetrievalRecord]] = {}
    for record in records:
        grouped.setdefault(record.query, []).append(record)
    return grouped


def mine_curriculum_examples(records: Iterable[RetrievalRecord]) -> Dict[str, List[RetrievalRecord]]:
    buckets = {
        "easy_pos": [],
        "hard_pos": [],
        "hard_neg": [],
        "token_sim_neg": [],
        "random_neg": [],
    }
    for record in records:
        if record.teacher_grade >= 3 and len(record.channels) >= 2 and record.lexical_overlap >= 0.8:
            record.bucket = "easy_pos"
            buckets["easy_pos"].append(record)
        elif record.teacher_grade >= 3:
            record.bucket = "hard_pos"
            buckets["hard_pos"].append(record)
        elif record.teacher_grade <= 1 and record.token_similarity >= 0.45:
            record.bucket = "token_sim_neg"
            buckets["token_sim_neg"].append(record)
        elif record.teacher_grade <= 1 and len(record.channels) == 1:
            record.bucket = "hard_neg"
            buckets["hard_neg"].append(record)
        else:
            record.bucket = "random_neg"
            buckets["random_neg"].append(record)
    return buckets


def curriculum_triplets(records: Iterable[RetrievalRecord]) -> Dict[str, List[tuple[RetrievalRecord, RetrievalRecord, RetrievalRecord]]]:
    grouped = group_by_query(records)
    stage2: List[tuple[RetrievalRecord, RetrievalRecord, RetrievalRecord]] = []
    stage3: List[tuple[RetrievalRecord, RetrievalRecord, RetrievalRecord]] = []
    for query_records in grouped.values():
        easy_pos = [r for r in query_records if r.bucket == "easy_pos"]
        hard_pos = [r for r in query_records if r.bucket == "hard_pos"]
        hard_neg = [r for r in query_records if r.bucket == "hard_neg"]
        token_sim_neg = [r for r in query_records if r.bucket == "token_sim_neg"]
        anchor_pool = easy_pos or hard_pos
        if anchor_pool and hard_pos and hard_neg:
            stage2.append((anchor_pool[0], hard_pos[0], hard_neg[0]))
        if hard_pos and token_sim_neg:
            stage3.append((hard_pos[0], hard_pos[0], token_sim_neg[0]))
    return {"stage2": stage2, "stage3": stage3}


def all_texts(records: Iterable[RetrievalRecord]) -> List[str]:
    texts: List[str] = []
    for record in records:
        texts.extend([record.query, record.item_title])
    return texts
