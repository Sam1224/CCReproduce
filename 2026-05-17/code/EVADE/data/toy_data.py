"""
Toy dataset generator for EVADE benchmark.
Simulates the structure of the real EVADE dataset:
  - 6 product categories
  - Text and image samples
  - Binary labels: 0 = compliant, 1 = evasive violation
"""

import random
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


CATEGORIES = [
    "body_shaping",
    "height_growth",
    "health_supplements",
    "medical_devices",
    "weight_loss",
    "skin_care",
]

# Policy rules per category (simplified for reproduction)
POLICY_RULES = {
    "body_shaping": [
        "Do not claim guaranteed weight loss or body shape changes",
        "Do not use before/after comparison without scientific evidence",
        "Do not claim results within specific short timeframes",
    ],
    "height_growth": [
        "Do not claim to increase height after growth plates are closed",
        "Do not use misleading statistics about height gain",
        "Do not target adults with height growth claims",
    ],
    "health_supplements": [
        "Do not make unverified medical or therapeutic claims",
        "Do not claim to cure, treat, or prevent diseases",
        "Do not reference clinical studies without proper citation",
    ],
    "medical_devices": [
        "Do not misrepresent device efficacy without clinical evidence",
        "Do not claim regulatory approval not obtained",
        "Do not use testimonials implying medical effectiveness",
    ],
    "weight_loss": [
        "Do not promise specific weight loss amounts without evidence",
        "Do not claim rapid results (e.g., lose 10kg in 1 week)",
        "Do not use misleading 'natural' or 'guaranteed' claims",
    ],
    "skin_care": [
        "Do not claim pharmaceutical-level efficacy without approval",
        "Do not use before/after images that are digitally altered",
        "Do not claim to permanently change skin structure",
    ],
}

# Evasive text patterns (subtle, non-obvious violations)
EVASIVE_TEXTS = {
    "body_shaping": [
        "用户反馈：穿上之后感觉整个人都不一样了，自信满满",  # Implicit shape claim via testimonial
        "科学塑形原理，帮助打造理想轮廓，让您的身材更有曲线",  # Vague "scientific" claim
        "长期坚持，很多用户都说效果明显，具体因人而异",  # Implied results without evidence
    ],
    "health_supplements": [
        "调节身体机能，增强免疫力，让您每天精力充沛",  # Health claim without evidence
        "采用传统配方，结合现代工艺，呵护您的健康",  # Vague efficacy claim
        "90%的用户在使用后反映身体状态明显改善",  # Unsupported statistic
    ],
    "height_growth": [
        "助力青少年成长，补充营养，让孩子长得更好",  # Height growth implication
        "成长营养配方，专为发育期设计，每天一杯助成长",  # Growth claim
    ],
}

COMPLIANT_TEXTS = {
    "body_shaping": [
        "这款运动服采用弹力面料，穿着舒适，适合日常运动",
        "高品质瑜伽裤，透气性好，适合各种运动场合",
    ],
    "health_supplements": [
        "维生素C片，每片含100mg维生素C，符合国家标准",
        "膳食纤维粉，原材料为天然燕麦，无添加防腐剂",
    ],
    "height_growth": [
        "儿童营养品，含有钙、铁、锌等多种矿物质",
        "均衡营养奶粉，适合3岁以上儿童日常饮用",
    ],
}


@dataclass
class EVADESample:
    sample_id: str
    category: str
    text: str
    image_path: Optional[str]  # None for text-only samples
    label: int  # 0 = compliant, 1 = evasive violation
    violation_type: Optional[str]  # Which policy rule is violated
    task_type: str  # "single_violation" or "all_in_one"


def generate_toy_dataset(
    n_samples: int = 100,
    task_type: str = "single_violation",
    seed: int = 42,
) -> list[EVADESample]:
    """Generate a toy EVADE dataset with the same interface as the real benchmark."""
    random.seed(seed)
    samples = []

    for i in range(n_samples):
        category = random.choice(CATEGORIES)
        is_evasive = random.random() > 0.5  # 50% evasive

        if is_evasive and category in EVASIVE_TEXTS:
            text = random.choice(EVASIVE_TEXTS[category])
            violation_type = random.choice(POLICY_RULES[category])
            label = 1
        else:
            text = random.choice(COMPLIANT_TEXTS.get(category, ["普通商品描述，无特殊功效声明"]))
            violation_type = None
            label = 0

        samples.append(EVADESample(
            sample_id=f"toy_{i:04d}",
            category=category,
            text=text,
            image_path=None,  # No real images in toy dataset
            label=label,
            violation_type=violation_type,
            task_type=task_type,
        ))

    return samples


def get_policy_prompt_single_violation(sample: EVADESample) -> str:
    """Build Single-Violation prompt: one rule, fine-grained reasoning."""
    rules = POLICY_RULES.get(sample.category, [])
    if not rules:
        rule = "Do not make unverified claims about product effects"
    else:
        rule = random.choice(rules)

    return (
        f"You are an e-commerce content moderator.\n"
        f"Product category: {sample.category.replace('_', ' ')}\n"
        f"Policy rule: {rule}\n\n"
        f"Product text:\n{sample.text}\n\n"
        f"Does this text violate the policy rule above? Answer with only 'violation' or 'compliant'."
    )


def get_policy_prompt_all_in_one(sample: EVADESample) -> str:
    """Build All-in-One prompt: all rules merged into unified instruction."""
    rules = POLICY_RULES.get(sample.category, [])
    rules_text = "\n".join(f"- {r}" for r in rules)

    return (
        f"You are an e-commerce content moderator.\n"
        f"Product category: {sample.category.replace('_', ' ')}\n"
        f"Platform policies (ALL must be followed):\n{rules_text}\n\n"
        f"Product text:\n{sample.text}\n\n"
        f"Does this text violate ANY of the policies above? "
        f"Answer with only 'violation' or 'compliant'."
    )
