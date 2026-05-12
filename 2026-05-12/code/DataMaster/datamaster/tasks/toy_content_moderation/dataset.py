from __future__ import annotations

import csv
import os
import random
from dataclasses import dataclass

from ...core.pool import DataPool
from ...core.state import Record


@dataclass
class ToyTaskSplits:
    pool: DataPool
    train_records: list[Record]
    val_records: list[Record]


def _mk_sentence(rng: random.Random, is_violation: bool) -> str:
    normal_templates = [
        "great quality {item} fast shipping",
        "authentic {brand} {item} discount today",
        "new arrival {brand} {item} limited offer",
        "recommended {item} for daily use",
        "unboxing video of {brand} {item}",
    ]
    violation_templates = [
        "free money click this link now",
        "100% guarantee fake brand {brand} {item}",
        "counterfeit {brand} {item} cheap price",
        "scam offer limited time free money",
    ]

    brands = ["nike", "adidas", "apple", "gucci", "chanel", "sony"]
    items = ["shoes", "bag", "phone", "headphones", "watch", "jacket"]

    tpl = rng.choice(violation_templates if is_violation else normal_templates)
    return tpl.format(brand=rng.choice(brands), item=rng.choice(items))


def _gen_source(rng: random.Random, name: str, n: int, violation_rate: float, label_noise: float) -> list[Record]:
    records: list[Record] = []
    for _ in range(n):
        is_violation = rng.random() < violation_rate
        text = _mk_sentence(rng, is_violation)
        label = 1 if is_violation else 0
        if rng.random() < label_noise:
            label = 1 - label
        records.append({"text": text, "label": label, "source": name})
    return records


def _save_csv(path: str, records: list[Record]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["text", "label", "source"])
        w.writeheader()
        for r in records:
            w.writerow({"text": r["text"], "label": int(r["label"]), "source": r["source"]})


def _load_csv(path: str) -> list[Record]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        out: list[Record] = []
        for row in r:
            out.append({"text": row["text"], "label": int(row["label"]), "source": row["source"]})
        return out


def build_toy_task(data_dir: str) -> ToyTaskSplits:
    os.makedirs(data_dir, exist_ok=True)

    base_path = os.path.join(data_dir, "base.csv")
    ext_clean_path = os.path.join(data_dir, "external_clean.csv")
    ext_noisy_path = os.path.join(data_dir, "external_noisy.csv")

    if not (os.path.exists(base_path) and os.path.exists(ext_clean_path) and os.path.exists(ext_noisy_path)):
        rng = random.Random(7)
        base = _gen_source(rng, "base", n=260, violation_rate=0.25, label_noise=0.18)
        ext_clean = _gen_source(rng, "external_clean", n=320, violation_rate=0.30, label_noise=0.03)
        ext_noisy = _gen_source(rng, "external_noisy", n=320, violation_rate=0.30, label_noise=0.35)
        _save_csv(base_path, base)
        _save_csv(ext_clean_path, ext_clean)
        _save_csv(ext_noisy_path, ext_noisy)

    base = _load_csv(base_path)
    ext_clean = _load_csv(ext_clean_path)
    ext_noisy = _load_csv(ext_noisy_path)

    pool = DataPool({"base": base, "external_clean": ext_clean, "external_noisy": ext_noisy})

    rng = random.Random(13)
    all_records = base + ext_clean + ext_noisy
    rng.shuffle(all_records)
    val = all_records[:240]
    train = all_records[240:]

    return ToyTaskSplits(pool=pool, train_records=train, val_records=val)
