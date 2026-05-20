"""Synthetic toy dataset for AgentIssueDiscovery.

Simulates a short-video platform corpus where:
- Most videos are benign or covered by existing policies
- A fraction contain 'emerging issues' not yet in the policy library
- Issues have variants (different presentation of same problem) and new sub-issues
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional


EXISTING_POLICIES = [
    "explicit violence",
    "hate speech",
    "spam promotion",
    "copyright infringement",
]

EMERGING_ISSUES = [
    # New issue A: subtle financial fraud via short video
    "subtle_financial_fraud",
    # New issue B: disguised tobacco advertising
    "disguised_tobacco_ad",
    # New issue C: misleading health claims
    "misleading_health_claims",
]

ISSUE_VARIANTS = {
    "subtle_financial_fraud": [
        "investment_tip_variant",
        "crypto_pump_variant",
        "lottery_scam_variant",
    ],
    "disguised_tobacco_ad": [
        "lifestyle_smoking_variant",
        "influencer_product_placement_variant",
    ],
    "misleading_health_claims": [
        "miracle_cure_variant",
        "diet_supplement_variant",
    ],
}

VIDEO_DESCRIPTIONS = {
    "benign": [
        "A cooking tutorial showing how to make pasta.",
        "A travel vlog visiting Paris landmarks.",
        "A fitness routine for beginners.",
        "A pet video featuring a playful golden retriever.",
        "A dance challenge video.",
    ],
    "subtle_financial_fraud": [
        "Someone sharing 'secret' investment tips that guarantee 300% returns.",
        "An influencer promoting a cryptocurrency with claims of overnight wealth.",
        "A video promising lottery-style winnings through a paid membership.",
    ],
    "disguised_tobacco_ad": [
        "A lifestyle video featuring cool characters casually smoking in a club setting.",
        "An influencer reviewing 'lifestyle accessories' that turn out to be tobacco products.",
    ],
    "misleading_health_claims": [
        "A video claiming a specific herb can cure cancer without any medical evidence.",
        "A fitness influencer promoting a supplement that 'dissolves fat overnight'.",
    ],
}


@dataclass
class SyntheticVideo:
    video_id: str
    description: str           # text description (simulates OCR + ASR)
    visual_tags: list[str]     # simulated visual content tags
    label: str                 # ground truth: 'benign', issue name, or variant name
    issue_type: Optional[str] = None  # 'existing', 'emerging_new', 'emerging_variant'
    parent_issue: Optional[str] = None  # for variants


def generate_synthetic_corpus(
    num_videos: int = 500,
    emerging_ratio: float = 0.20,
    variant_ratio: float = 0.60,
    seed: int = 42,
) -> list[SyntheticVideo]:
    """Generate a synthetic corpus of short videos.

    Args:
        num_videos: Total number of videos.
        emerging_ratio: Fraction of videos with emerging (unlisted) issues.
        variant_ratio: Among emerging, fraction that are variants of same issue.
        seed: Random seed.

    Returns:
        List of SyntheticVideo instances.
    """
    random.seed(seed)
    corpus: list[SyntheticVideo] = []

    num_emerging = int(num_videos * emerging_ratio)
    num_existing_issues = int(num_videos * 0.10)
    num_benign = num_videos - num_emerging - num_existing_issues

    # Benign videos
    for i in range(num_benign):
        desc = random.choice(VIDEO_DESCRIPTIONS["benign"])
        corpus.append(SyntheticVideo(
            video_id=f"v_benign_{i:04d}",
            description=desc,
            visual_tags=["normal_content", random.choice(["indoor", "outdoor"])],
            label="benign",
            issue_type=None,
        ))

    # Existing policy violations
    for i in range(num_existing_issues):
        policy = random.choice(EXISTING_POLICIES)
        corpus.append(SyntheticVideo(
            video_id=f"v_existing_{i:04d}",
            description=f"Video with {policy} content.",
            visual_tags=["flagged", policy.replace(" ", "_")],
            label=policy,
            issue_type="existing",
        ))

    # Emerging issues
    num_new_issues = int(num_emerging * (1 - variant_ratio))
    num_variants = num_emerging - num_new_issues

    for i in range(num_new_issues):
        issue = random.choice(EMERGING_ISSUES)
        desc = random.choice(VIDEO_DESCRIPTIONS[issue])
        corpus.append(SyntheticVideo(
            video_id=f"v_emerging_new_{i:04d}",
            description=desc,
            visual_tags=["subtle", issue],
            label=issue,
            issue_type="emerging_new",
        ))

    for i in range(num_variants):
        issue = random.choice(EMERGING_ISSUES)
        variant = random.choice(ISSUE_VARIANTS[issue])
        desc = random.choice(VIDEO_DESCRIPTIONS[issue])
        corpus.append(SyntheticVideo(
            video_id=f"v_variant_{i:04d}",
            description=desc + f" [variant: {variant}]",
            visual_tags=["variant", variant],
            label=variant,
            issue_type="emerging_variant",
            parent_issue=issue,
        ))

    random.shuffle(corpus)
    return corpus
