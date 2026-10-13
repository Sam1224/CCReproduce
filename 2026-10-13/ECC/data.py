import math
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


ATTR_SPACE = {
    "background_tone": ["cool", "warm", "neutral"],
    "copy_density": ["low", "medium", "high"],
    "visual_style": ["clean", "luxury", "busy"],
    "product_position": ["center", "left", "right"],
    "scene_type": ["studio", "lifestyle", "promo"],
}

GOOD_ATTRS = {
    "background_tone": {"warm": 1.0, "neutral": 0.6, "cool": 0.1},
    "copy_density": {"low": 1.0, "medium": 0.6, "high": 0.1},
    "visual_style": {"clean": 1.0, "luxury": 0.8, "busy": 0.1},
    "product_position": {"center": 1.0, "left": 0.4, "right": 0.4},
    "scene_type": {"lifestyle": 1.0, "studio": 0.7, "promo": 0.3},
}


@dataclass
class CreativeRecord:
    product_id: int
    category: str
    attrs: Dict[str, str]
    ctr: float


def score_attrs(attrs: Dict[str, str]) -> float:
    base = 0.0
    for name, value in attrs.items():
        base += GOOD_ATTRS[name][value]
    synergy = 0.0
    if attrs["scene_type"] == "lifestyle" and attrs["visual_style"] == "clean":
        synergy += 0.35
    if attrs["background_tone"] == "warm" and attrs["copy_density"] == "low":
        synergy += 0.25
    if attrs["product_position"] == "center" and attrs["visual_style"] == "busy":
        synergy -= 0.2
    return base + synergy


def sample_attrs(rng: random.Random) -> Dict[str, str]:
    return {name: rng.choice(values) for name, values in ATTR_SPACE.items()}


def generate_creatives(num_products: int = 120, creatives_per_product: int = 6, seed: int = 7) -> List[CreativeRecord]:
    rng = random.Random(seed)
    categories = ["beauty", "electronics", "home", "fashion"]
    records: List[CreativeRecord] = []
    for product_id in range(num_products):
        category = rng.choice(categories)
        for _ in range(creatives_per_product):
            attrs = sample_attrs(rng)
            ctr = score_attrs(attrs) + rng.uniform(-0.25, 0.25)
            records.append(
                CreativeRecord(
                    product_id=product_id,
                    category=category,
                    attrs=attrs,
                    ctr=ctr,
                )
            )
    return records


def normalize_ctr_within_product(records: Sequence[CreativeRecord]) -> List[Tuple[CreativeRecord, float]]:
    grouped: Dict[int, List[CreativeRecord]] = {}
    for item in records:
        grouped.setdefault(item.product_id, []).append(item)

    normalized: List[Tuple[CreativeRecord, float]] = []
    for product_records in grouped.values():
        ctrs = [item.ctr for item in product_records]
        mean = sum(ctrs) / len(ctrs)
        variance = sum((value - mean) ** 2 for value in ctrs) / max(1, len(ctrs) - 1)
        std = math.sqrt(variance) if variance > 1e-6 else 1.0
        for item in product_records:
            normalized.append((item, (item.ctr - mean) / std))
    return normalized


def mine_ctr_sensitive_attributes(records: Sequence[CreativeRecord], top_k: int = 4) -> List[str]:
    normalized = normalize_ctr_within_product(records)
    scores: List[Tuple[str, float]] = []
    for attr_name in ATTR_SPACE:
        grouped: Dict[str, List[float]] = {}
        for item, norm_ctr in normalized:
            grouped.setdefault(item.attrs[attr_name], []).append(norm_ctr)
        group_means = []
        for values in grouped.values():
            group_means.append(sum(values) / len(values))
        mean_of_means = sum(group_means) / len(group_means)
        between = sum((value - mean_of_means) ** 2 for value in group_means)
        within = 0.0
        for values in grouped.values():
            inner_mean = sum(values) / len(values)
            within += sum((value - inner_mean) ** 2 for value in values)
        effect = between / max(within, 1e-6)
        scores.append((attr_name, effect))
    scores.sort(key=lambda item: item[1], reverse=True)
    return [name for name, _ in scores[:top_k]]


def diff_attrs(source: Dict[str, str], target: Dict[str, str], selected_attrs: Sequence[str]) -> List[str]:
    operations = []
    for attr_name in selected_attrs:
        if source[attr_name] != target[attr_name]:
            operations.append("change {} to {}".format(attr_name, target[attr_name]))
    return operations


def creative_to_prompt(record: CreativeRecord, selected_attrs: Sequence[str]) -> str:
    attrs_text = " ".join("{}={}".format(name, record.attrs[name]) for name in selected_attrs)
    return "product={} category={} {}".format(record.product_id, record.category, attrs_text)


def build_sft_examples(records: Sequence[CreativeRecord], selected_attrs: Sequence[str]) -> List[Tuple[str, str]]:
    grouped: Dict[int, List[CreativeRecord]] = {}
    for item in records:
        grouped.setdefault(item.product_id, []).append(item)

    examples: List[Tuple[str, str]] = []
    for product_records in grouped.values():
        ordered = sorted(product_records, key=lambda item: item.ctr)
        low = ordered[0]
        high = ordered[-1]
        operations = diff_attrs(low.attrs, high.attrs, selected_attrs)
        if not operations:
            continue
        prompt = creative_to_prompt(low, selected_attrs)
        target = " ; ".join(operations)
        examples.append((prompt, target))
    return examples


def shuffle_equivalent_suggestion(text: str, seed: int) -> str:
    ops = [chunk.strip() for chunk in text.split(";") if chunk.strip()]
    rng = random.Random(seed)
    rng.shuffle(ops)
    return " ; ".join(ops)


def corrupt_suggestion(text: str, selected_attrs: Sequence[str], seed: int) -> str:
    ops = [chunk.strip() for chunk in text.split(";") if chunk.strip()]
    rng = random.Random(seed)
    if not ops:
        return text
    replace_index = rng.randrange(len(ops))
    random_attr = rng.choice(list(selected_attrs))
    random_value = rng.choice(ATTR_SPACE[random_attr])
    ops[replace_index] = "change {} to {}".format(random_attr, random_value)
    return " ; ".join(ops)


def build_preference_examples(records: Sequence[CreativeRecord], selected_attrs: Sequence[str]) -> List[Tuple[str, str, str, str]]:
    sft_examples = build_sft_examples(records, selected_attrs)
    preference_rows = []
    for index, (prompt, chosen) in enumerate(sft_examples):
        chosen_shuffled = shuffle_equivalent_suggestion(chosen, seed=index + 97)
        rejected = corrupt_suggestion(chosen, selected_attrs, seed=index + 199)
        preference_rows.append((prompt, chosen, chosen_shuffled, rejected))
    return preference_rows


def split_train_valid(items: Sequence, ratio: float = 0.85) -> Tuple[List, List]:
    boundary = int(len(items) * ratio)
    return list(items[:boundary]), list(items[boundary:])
