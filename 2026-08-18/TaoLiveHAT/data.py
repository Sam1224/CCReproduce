import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset


INTENTS = [
    "product_answer",
    "logistics_answer",
    "compliance_refuse",
    "campaign_guidance",
]

INTENT_TO_LABEL = {name: idx for idx, name in enumerate(INTENTS)}

SEMANTIC_PROTOTYPES: Dict[str, Sequence[float]] = {
    "product_answer": [1.2, 0.9, 0.3, 0.1, 0.5, 0.0, -0.3, 0.2],
    "logistics_answer": [0.2, 1.1, 0.6, -0.2, 0.3, 0.8, -0.1, 0.0],
    "compliance_refuse": [-0.8, -0.2, 1.3, 0.9, 0.2, -0.1, 0.6, 0.5],
    "campaign_guidance": [0.7, 0.1, 0.0, 0.4, 1.2, 0.5, 0.1, -0.2],
}

PROMPT_PROTOTYPES: Dict[str, Sequence[float]] = {
    "product_answer": [1.0, 0.2, 0.0, 0.1],
    "logistics_answer": [0.2, 1.0, 0.0, 0.1],
    "compliance_refuse": [0.0, 0.2, 1.0, 0.6],
    "campaign_guidance": [0.3, 0.2, 0.2, 1.1],
}

TOOL_PROTOTYPES: Dict[str, Sequence[float]] = {
    "product_answer": [1.0, 0.1, 0.0, 0.2],
    "logistics_answer": [0.2, 1.1, 0.0, 0.0],
    "compliance_refuse": [0.1, 0.0, 1.2, 0.3],
    "campaign_guidance": [0.2, 0.1, 0.2, 1.0],
}

HOOK_PROTOTYPES: Dict[str, Sequence[float]] = {
    "product_answer": [0.8, 0.3, 0.0, 0.2],
    "logistics_answer": [0.1, 0.9, 0.0, 0.2],
    "compliance_refuse": [0.0, 0.2, 1.2, 0.7],
    "campaign_guidance": [0.2, 0.0, 0.3, 1.0],
}

TASK_TEMPLATES = {
    "product_answer": [
        "这件外套防水吗",
        "主播，这个鞋子的尺码偏大还是偏小",
        "面料会不会起球",
    ],
    "logistics_answer": [
        "今天下单多久可以送到杭州",
        "支持七天无理由退货吗",
        "海外仓发货需要多久",
    ],
    "compliance_refuse": [
        "能不能保证治百病",
        "帮我夸大一下减肥效果",
        "把禁售承诺说得再狠一点",
    ],
    "campaign_guidance": [
        "满减活动怎么凑单最划算",
        "今晚直播间的优惠券怎么领",
        "新人礼和店铺券能叠加吗",
    ],
}


@dataclass
class Sample:
    features: List[float]
    label: int
    intent: str
    utterance: str
    variant_level: float


class HarnessDataset(Dataset):
    def __init__(self, samples: List[Sample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        return {
            "x": torch.tensor(item.features, dtype=torch.float32),
            "y": torch.tensor(item.label, dtype=torch.long),
            "intent": item.intent,
            "utterance": item.utterance,
            "variant_level": torch.tensor(item.variant_level, dtype=torch.float32),
        }


def feature_dim() -> int:
    return 20


def _jitter(base: Sequence[float], rng: random.Random, scale: float) -> List[float]:
    return [float(v + rng.uniform(-scale, scale)) for v in base]


def _variantize(base: Sequence[float], rng: random.Random, scale: float, mix_bias: float) -> List[float]:
    mixed = []
    for i, value in enumerate(base):
        swapped = base[(i + 1) % len(base)]
        mixed.append(float((1 - mix_bias) * value + mix_bias * swapped + rng.uniform(-scale, scale)))
    return mixed


def build_samples(
    *,
    size: int,
    seed: int,
    augment_harness: bool,
    variant_split: bool = False,
) -> List[Sample]:
    rng = random.Random(seed)
    samples: List[Sample] = []

    intents = list(INTENTS)
    for index in range(size):
        intent = intents[index % len(intents)]
        utterance = rng.choice(TASK_TEMPLATES[intent])

        neighbor_intent = intents[(INTENT_TO_LABEL[intent] + 1) % len(intents)]
        if variant_split:
            semantic = [
                float(
                    0.58 * SEMANTIC_PROTOTYPES[intent][i]
                    + 0.42 * SEMANTIC_PROTOTYPES[neighbor_intent][i]
                    + rng.uniform(-0.22, 0.22)
                )
                for i in range(8)
            ]
        elif augment_harness:
            semantic = [
                float(
                    0.82 * SEMANTIC_PROTOTYPES[intent][i]
                    + 0.18 * SEMANTIC_PROTOTYPES[neighbor_intent][i]
                    + rng.uniform(-0.18, 0.18)
                )
                for i in range(8)
            ]
        else:
            semantic = _jitter(SEMANTIC_PROTOTYPES[intent], rng, 0.20)

        if augment_harness or variant_split:
            prompt = _variantize(PROMPT_PROTOTYPES[intent], rng, 0.35, 0.18 if augment_harness else 0.26)
            tool = _variantize(TOOL_PROTOTYPES[intent], rng, 0.35, 0.20 if augment_harness else 0.28)
            hook = _variantize(HOOK_PROTOTYPES[intent], rng, 0.30, 0.15 if augment_harness else 0.30)
            variant_level = 0.55 if augment_harness else 0.85
        else:
            prompt = _jitter(PROMPT_PROTOTYPES[intent], rng, 0.10)
            tool = _jitter(TOOL_PROTOTYPES[intent], rng, 0.10)
            hook = _jitter(HOOK_PROTOTYPES[intent], rng, 0.10)
            variant_level = 0.05

        noise_scalar = rng.uniform(-0.15, 0.15)
        route_hint = [
            float((INTENT_TO_LABEL[intent] / (len(INTENTS) - 1)) + noise_scalar),
            float(variant_level + rng.uniform(-0.05, 0.05)),
            float(1.0 if intent == "compliance_refuse" else 0.0),
            float(1.0 if intent in {"product_answer", "campaign_guidance"} else 0.0),
        ]
        features = semantic + prompt + tool + hook + route_hint
        samples.append(
            Sample(
                features=features,
                label=INTENT_TO_LABEL[intent],
                intent=intent,
                utterance=utterance,
                variant_level=variant_level,
            )
        )
    rng.shuffle(samples)
    return samples


def make_splits(train_size: int = 1200, val_size: int = 240, test_size: int = 320):
    fixed_train = HarnessDataset(build_samples(size=train_size, seed=13, augment_harness=False))
    hat_train = HarnessDataset(build_samples(size=train_size, seed=29, augment_harness=True))
    val = HarnessDataset(build_samples(size=val_size, seed=41, augment_harness=True))
    base_test = HarnessDataset(build_samples(size=test_size, seed=53, augment_harness=False))
    variant_test = HarnessDataset(build_samples(size=test_size, seed=67, augment_harness=False, variant_split=True))
    return {
        "fixed_train": fixed_train,
        "hat_train": hat_train,
        "val": val,
        "base_test": base_test,
        "variant_test": variant_test,
    }


def benchmark_cases() -> List[Tuple[str, str]]:
    rows = []
    for intent in INTENTS:
        rows.append((intent, TASK_TEMPLATES[intent][0]))
    return rows
