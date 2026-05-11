"""
ARGUS — Data utilities
Toy dataset interface aligned to the paper's ad governance setup.

Paper: "ARGUS: Policy-Adaptive Ad Governance via Evolving Reinforcement
        with Adversarial Umpiring" (arXiv 2605.02200)

Ad governance dataset:
  - Each sample: (ad_image_path, ad_text, label, policy_version)
  - label ∈ {0: compliant, 1: violation}
  - policy_version tracks which regulatory mandate applies
"""

import os
import json
import random
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


@dataclass
class AdSample:
    """Single ad governance sample."""
    ad_id: str
    image_path: Optional[str]
    ad_text: str
    label: int                   # 0 = compliant, 1 = violation
    policy_version: str          # e.g. "v1_baseline", "v2_edu_anxiety"
    is_gray_area: bool = False   # ambiguous / borderline samples


class AdGovernanceDataset(Dataset):
    """
    Ad governance dataset.

    Toy data is generated synthetically for interface demonstration.
    In production, replace with actual annotated ad samples.
    """

    # Toy ad templates for different policy scenarios
    _COMPLIANT_TEMPLATES = [
        "Limited time offer! Get 20% off our premium products.",
        "Free shipping on orders over $50. Shop now!",
        "New arrivals this season. Explore our collection.",
        "High quality products at affordable prices.",
        "Customer satisfaction guaranteed. Easy returns.",
    ]

    _VIOLATION_TEMPLATES = {
        "v2_edu_anxiety": [
            "Your child will fall behind without this course! Act now!",
            "Don't let your kid miss out — other children are already ahead!",
            "Worried your child isn't keeping up? Our tutoring fixes that.",
        ],
        "v3_appearance_anxiety": [
            "Fix your ugly skin problems with our cream!",
            "Tired of looking fat? Our product transforms you in 7 days!",
            "Don't let your appearance hold you back. Get beautiful now!",
        ],
        "v1_baseline": [
            "Buy now or regret forever! Stock running out!",
            "100% guaranteed weight loss in 3 days or money back!",
            "Doctor recommended (results may vary) — proven effective!",
        ],
    }

    _GRAY_AREA_TEMPLATES = [
        "Help your child reach their full potential this semester.",
        "Skin care tips that professionals use — now available to you.",
        "Limited stock remaining. Don't miss this opportunity.",
    ]

    def __init__(
        self,
        policy_version: str = "v2_edu_anxiety",
        split: str = "train",
        num_samples: int = 500,
        include_gray_area: bool = True,
        image_size: int = 224,
        seed: int = 42,
    ):
        self.policy_version = policy_version
        self.split = split
        self.image_size = image_size
        random.seed(seed + hash(split) % 1000)

        self.samples = self._generate_toy_samples(
            num_samples, include_gray_area
        )

    def _generate_toy_samples(
        self, num_samples: int, include_gray_area: bool
    ) -> List[AdSample]:
        samples = []
        violation_texts = self._VIOLATION_TEMPLATES.get(
            self.policy_version,
            self._VIOLATION_TEMPLATES["v1_baseline"]
        )

        for i in range(num_samples):
            r = random.random()
            if r < 0.4:
                label, text, gray = 0, random.choice(self._COMPLIANT_TEMPLATES), False
            elif r < 0.7:
                label, text, gray = 1, random.choice(violation_texts), False
            elif include_gray_area and r < 0.9:
                label = random.choice([0, 1])
                text = random.choice(self._GRAY_AREA_TEMPLATES)
                gray = True
            else:
                label, text, gray = 0, random.choice(self._COMPLIANT_TEMPLATES), False

            samples.append(AdSample(
                ad_id=f"ad_{i:06d}",
                image_path=None,       # no real images in toy mode
                ad_text=text,
                label=label,
                policy_version=self.policy_version,
                is_gray_area=gray,
            ))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        # Synthetic image tensor (random noise as toy image)
        image_tensor = torch.randn(3, self.image_size, self.image_size)

        return {
            "ad_id": sample.ad_id,
            "image": image_tensor,
            "text": sample.ad_text,
            "label": torch.tensor(sample.label, dtype=torch.long),
            "policy_version": sample.policy_version,
            "is_gray_area": sample.is_gray_area,
        }


class PolicyKnowledgeBase:
    """
    RAG-based policy knowledge base.

    Stores policy clauses and supports dense retrieval for the
    Umpire VLM's RAG-enhanced policy reasoning.

    Paper §3.2: "a neutral Umpire VLM then adjudicates ... incorporating
    RAG-enhanced policy knowledge"
    """

    def __init__(self, policy_version: str = "v2_edu_anxiety"):
        self.policy_version = policy_version
        self.clauses = self._load_toy_policy(policy_version)
        # In production: encode clauses with a sentence encoder for FAISS retrieval
        self._embeddings: Optional[np.ndarray] = None

    def _load_toy_policy(self, version: str) -> List[Dict[str, str]]:
        """Toy policy clauses — replace with actual regulatory documents."""
        base_clauses = [
            {"id": "c001", "text": "Advertisements must not make false or misleading claims about product efficacy."},
            {"id": "c002", "text": "Claims must be substantiated by verifiable evidence from credible sources."},
            {"id": "c003", "text": "Urgency tactics (e.g., 'act now or regret') must not exploit consumer fear."},
        ]
        version_specific = {
            "v2_edu_anxiety": [
                {"id": "c101", "text": "Educational ads must not induce anxiety about a child's academic performance or future."},
                {"id": "c102", "text": "Comparisons implying a child is falling behind peers are prohibited."},
                {"id": "c103", "text": "Ads targeting parents must not leverage parental guilt as a persuasion mechanism."},
            ],
            "v3_appearance_anxiety": [
                {"id": "c201", "text": "Ads must not use language implying a consumer's appearance is defective or inferior."},
                {"id": "c202", "text": "Before/after claims must be accompanied by evidence and appropriate disclaimers."},
            ],
        }
        return base_clauses + version_specific.get(version, [])

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, str]]:
        """
        Simple keyword-based retrieval for toy mode.
        Production: replace with FAISS dense retrieval over sentence embeddings.
        """
        query_lower = query.lower()
        scored = []
        for clause in self.clauses:
            overlap = sum(
                word in clause["text"].lower()
                for word in query_lower.split()
            )
            scored.append((overlap, clause))
        scored.sort(key=lambda x: -x[0])
        return [c for _, c in scored[:top_k]]

    def format_for_prompt(self, retrieved: List[Dict[str, str]]) -> str:
        lines = ["Relevant Policy Clauses:"]
        for i, clause in enumerate(retrieved, 1):
            lines.append(f"  [{clause['id']}] {clause['text']}")
        return "\n".join(lines)
