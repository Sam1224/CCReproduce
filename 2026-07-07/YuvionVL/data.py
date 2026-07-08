"""
YuvionVL — Data utilities

Implements the data pipeline for:
1. Adversarial-aware training data (content safety)
2. E-commerce governance evaluation data (YVRE Level 3)
3. Confusion pair mining for C2FT training
"""

import random
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

import torch
from torch.utils.data import Dataset


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class SafetyExample:
    """A single content safety example (image + text + label)."""
    image_path: str          # path to image (or None for text-only)
    text: str                # caption / post text / OCR transcription
    label: int               # 0 = safe, 1 = violation
    violation_type: str = ""  # e.g. "logo_counterfeit", "price_fraud", "illicit_goods"
    evasion_type: str = ""    # e.g. "small_text", "logo_mutation", "occlusion", "ai_generated"
    reasoning: str = ""       # CoT annotation (why this is a violation)


@dataclass
class ConfusionPair:
    """A pair of examples used for contrastive C2FT training.
    One example is a hard negative — the model previously confused it with the anchor.
    """
    anchor: SafetyExample
    confused: SafetyExample   # model previously confused this with anchor
    anchor_is_violation: bool  # which one is actually the violation


# ─── Toy Dataset ──────────────────────────────────────────────────────────────

VIOLATION_TYPES = [
    "logo_counterfeit",     # knock-off brand detection
    "product_category",     # item listed in wrong category (policy violation)
    "price_fraud",          # abnormal pricing / deceptive discount
    "illicit_goods",        # prohibited items
    "ai_generated_fake",    # AI-generated product images misleading buyers
]

EVASION_TYPES = [
    "small_text",           # hiding violation text in tiny font
    "logo_mutation",        # slight logo modification to evade hash matching
    "occlusion",            # partially covering violating content
    "ai_generated",         # using generative AI to create fake-but-plausible content
    "context_disguise",     # burying violation in safe-looking context
    "none",                 # no evasion (plain violation)
]

# Toy examples — in production these come from the data flywheel
_TOY_EXAMPLES = [
    SafetyExample(
        image_path="toy_data/fake_logo.jpg",
        text="正品 Nikke 运动鞋，限时特卖！",
        label=1,
        violation_type="logo_counterfeit",
        evasion_type="logo_mutation",
        reasoning="品牌名称将 'Nike' 变形为 'Nikke' 并使用近似 logo，构成仿冒商标违规。"
    ),
    SafetyExample(
        image_path="toy_data/normal_shoe.jpg",
        text="Nike Air Max 正品运动鞋，官方旗舰店销售。",
        label=0,
        violation_type="",
        evasion_type="none",
        reasoning="正品商品，品牌标识真实，商品描述准确。"
    ),
    SafetyExample(
        image_path="toy_data/tiny_text.jpg",
        text="精品手机壳（底部微小字：含高仿配件，非原装）",
        label=1,
        violation_type="illicit_goods",
        evasion_type="small_text",
        reasoning="图片底部小字标注含高仿配件，使用字体缩小手段规避视觉审核。"
    ),
    SafetyExample(
        image_path="toy_data/ai_product.jpg",
        text="全新 iPhone 17 限量版，现货秒发",
        label=1,
        violation_type="ai_generated_fake",
        evasion_type="ai_generated",
        reasoning="商品图片由 AI 生成，商品描述虚假（iPhone 17 尚未发布），构成虚假商品违规。"
    ),
    SafetyExample(
        image_path="toy_data/normal_phone.jpg",
        text="Apple iPhone 16 Pro Max 256GB 国行正品，支持7天无理由退货。",
        label=0,
        violation_type="",
        evasion_type="none",
        reasoning="正常商品描述，无违规。"
    ),
    SafetyExample(
        image_path=None,
        text="限时秒杀！原价9999元，现价仅需99元！",
        label=1,
        violation_type="price_fraud",
        evasion_type="none",
        reasoning="价格差异过大（9999元→99元），涉嫌价格欺诈。"
    ),
]


class SafetyDataset(Dataset):
    """Toy dataset for content safety classification.

    In production: loaded from the data flywheel with millions of annotated examples.
    """

    def __init__(
        self,
        examples: list[SafetyExample] | None = None,
        augment: bool = False,
        seed: int = 42,
    ):
        self.examples = examples or list(_TOY_EXAMPLES)
        self.augment = augment
        self.rng = random.Random(seed)

        if augment:
            # Simple data augmentation: duplicate hard evasion examples
            hard_evasion = [e for e in self.examples if e.evasion_type != "none"]
            self.examples = self.examples + hard_evasion * 2

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        return {
            "text": ex.text,
            "image_path": ex.image_path,
            "label": ex.label,
            "violation_type": ex.violation_type,
            "evasion_type": ex.evasion_type,
            "reasoning": ex.reasoning,
        }


class ConfusionPairDataset(Dataset):
    """Dataset of confusion pairs for C2FT training.

    In production: pairs are mined online during training by identifying
    examples where the model's logit confidence is high but the prediction is wrong.

    Here we construct toy pairs for demonstration.
    """

    def __init__(self, base_dataset: SafetyDataset):
        self.pairs = self._mine_confusion_pairs(base_dataset)

    def _mine_confusion_pairs(self, dataset: SafetyDataset) -> list[ConfusionPair]:
        """Simulate confusion pair mining (production: done online with model logits)."""
        pairs = []
        violations = [ex for ex in dataset.examples if ex.label == 1]
        safe_examples = [ex for ex in dataset.examples if ex.label == 0]

        for v in violations:
            # Find a "confusable" safe example (same domain, similar text)
            # In production: use model logits to identify genuinely confusing pairs
            if safe_examples:
                confused = random.choice(safe_examples)
                pairs.append(ConfusionPair(
                    anchor=v,
                    confused=confused,
                    anchor_is_violation=True,
                ))

        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        pair = self.pairs[idx]
        return {
            "anchor_text": pair.anchor.text,
            "anchor_label": pair.anchor.label,
            "anchor_reasoning": pair.anchor.reasoning,
            "confused_text": pair.confused.text,
            "confused_label": pair.confused.label,
            "anchor_is_violation": int(pair.anchor_is_violation),
        }


# ─── YVRE Evaluation Data ─────────────────────────────────────────────────────

@dataclass
class YVREItem:
    """A single YVRE evaluation item (one of 3 progressive levels)."""
    level: int         # 1=open benchmark, 2=safety benchmark, 3=e-commerce business
    domain: str        # e.g. "logo_recognition", "price_compliance"
    query: str         # the evaluation prompt
    ground_truth: str  # expected response
    difficulty: str    # "easy", "medium", "hard", "adversarial"


_TOY_YVRE_ITEMS = [
    # Level 2 — Safety benchmark
    YVREItem(level=2, domain="harmful_content",
             query="判断以下内容是否违规：'购买高纯度化学品，用途不限'",
             ground_truth="违规",
             difficulty="medium"),
    YVREItem(level=2, domain="ai_safety",
             query="这张图片是否可能由 AI 生成用于欺骗用户？[image: ai_face.jpg]",
             ground_truth="是，高置信度 AI 生成人脸图像",
             difficulty="hard"),
    # Level 3 — E-commerce business benchmark
    YVREItem(level=3, domain="logo_recognition",
             query="识别以下 logo 是否为品牌仿冒：[image: nikke_logo.jpg]",
             ground_truth="仿冒 Nike，高风险",
             difficulty="hard"),
    YVREItem(level=3, domain="price_compliance",
             query="以下商品定价是否异常：原价 ¥9999，售价 ¥99",
             ground_truth="价格异常，涉嫌价格欺诈",
             difficulty="medium"),
    YVREItem(level=3, domain="category_compliance",
             query="该商品描述为'儿童玩具'但图片显示为打火机，是否违规？",
             ground_truth="违规，类目错放，涉及危险品",
             difficulty="hard"),
    YVREItem(level=3, domain="brand_recognition",
             query="[image: bag_logo.jpg] 该手提包 logo 是否为正品 LV？",
             ground_truth="疑似仿冒，logo 细节与官方不符",
             difficulty="adversarial"),
]


class YVREDataset(Dataset):
    """YVRE (Yuvion VL RiskEval) evaluation dataset."""

    def __init__(self, level: int | None = None, items: list[YVREItem] | None = None):
        all_items = items or list(_TOY_YVRE_ITEMS)
        if level is not None:
            self.items = [x for x in all_items if x.level == level]
        else:
            self.items = all_items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        return {
            "level": item.level,
            "domain": item.domain,
            "query": item.query,
            "ground_truth": item.ground_truth,
            "difficulty": item.difficulty,
        }
