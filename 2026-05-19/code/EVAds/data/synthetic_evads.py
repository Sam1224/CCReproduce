"""Synthetic E-VAds dataset.

Simulates e-commerce short video Q&A pairs with commercial intent annotations.
Real E-VAds uses 3,961 Taobao videos + 19,785 open-ended Q&A pairs.

Question types:
  1. Product identification (what product is shown?)
  2. Commercial intent (what is the creator trying to sell/promote?)
  3. Audience value (what value does this content offer to the viewer?)
  4. Marketing strategy (what marketing technique is used?)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional


PRODUCT_CATEGORIES = [
    "skincare", "fashion", "electronics", "food & beverage",
    "home appliances", "sports equipment", "books", "toys",
]

COMMERCIAL_INTENTS = [
    "direct product promotion",
    "affiliate marketing",
    "brand awareness",
    "limited-time offer push",
    "lifestyle endorsement",
    "tutorial with product placement",
    "unboxing/review",
]

MARKETING_TECHNIQUES = [
    "emotional appeal",
    "scarcity/urgency",
    "social proof (reviews)",
    "celebrity/KOL endorsement",
    "before-after demonstration",
    "price comparison",
    "discount announcement",
]


@dataclass
class EComVideoQA:
    video_id: str
    product_category: str
    video_description: str   # simulated video content description
    questions: list[dict]    # list of {question, answer, question_type}
    commercial_intent: str
    marketing_technique: str
    density_score: float     # simulated multimodal info density (0-1)


def _generate_video_desc(category: str, intent: str, technique: str) -> str:
    templates = [
        f"A {category} influencer demonstrates product usage while using {technique}.",
        f"Short video showcasing {category} product with {intent} as primary goal.",
        f"Creator uses {technique} to promote {category} items in a 60-second clip.",
        f"E-commerce livestream highlight: {category} product with {intent}.",
    ]
    return random.choice(templates)


def _generate_qas(category: str, intent: str, technique: str) -> list[dict]:
    return [
        {
            "question": f"What product category is being promoted in this video?",
            "answer": category,
            "question_type": "product_identification",
        },
        {
            "question": "What is the creator's primary commercial intent in this video?",
            "answer": intent,
            "question_type": "commercial_intent",
        },
        {
            "question": "What marketing technique does the creator primarily use?",
            "answer": technique,
            "question_type": "marketing_strategy",
        },
        {
            "question": "What value does this video offer to its target audience?",
            "answer": f"Information about {category} products combined with {technique}.",
            "question_type": "audience_value",
        },
        {
            "question": "Is this video primarily for entertainment or commercial purposes?",
            "answer": "commercial" if intent != "lifestyle endorsement" else "mixed",
            "question_type": "commercial_intent",
        },
    ]


def generate_evads_dataset(
    num_videos: int = 200,
    seed: int = 42,
) -> list[EComVideoQA]:
    random.seed(seed)
    dataset = []

    for i in range(num_videos):
        category = random.choice(PRODUCT_CATEGORIES)
        intent = random.choice(COMMERCIAL_INTENTS)
        technique = random.choice(MARKETING_TECHNIQUES)

        video = EComVideoQA(
            video_id=f"evads_toy_{i:04d}",
            product_category=category,
            video_description=_generate_video_desc(category, intent, technique),
            questions=_generate_qas(category, intent, technique),
            commercial_intent=intent,
            marketing_technique=technique,
            density_score=random.uniform(0.6, 1.0),  # e-commerce = high density
        )
        dataset.append(video)

    return dataset


if __name__ == "__main__":
    dataset = generate_evads_dataset(num_videos=200)
    print(f"Generated {len(dataset)} E-VAds toy samples")
    print(f"Sample video: {dataset[0].video_id}")
    print(f"  Category: {dataset[0].product_category}")
    print(f"  Intent: {dataset[0].commercial_intent}")
    print(f"  First Q: {dataset[0].questions[0]['question']}")
    print(f"  First A: {dataset[0].questions[0]['answer']}")
    print(f"  Density: {dataset[0].density_score:.3f}")
