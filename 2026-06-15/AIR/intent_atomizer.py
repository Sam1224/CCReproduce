"""
AIR - Intent Atomizer (Offline LLM Phase)

Implements offline intent decomposition from user content behaviors into
'intent atoms' — fine-grained, indexable semantic units.

Paper: https://arxiv.org/abs/2606.10357 (Section 3.1 - Intent Atomization)

The paper uses a large LLM to extract:
  1. Fine-grained interest atoms  (e.g., "user interested in high-protein recipes")
  2. Implicit purchase intent atoms (e.g., "user may want kitchen appliances")

We simulate this with a lightweight keyword extraction + template approach for toy.
"""
import json
import os
import re
from collections import Counter
from typing import Optional

import numpy as np


# ── Intent atom templates (simulating LLM extraction) ──────────────────────
INTEREST_TEMPLATES = {
    "cooking": [
        "interested in {action} cooking content",
        "engages with culinary videos frequently",
        "potential buyer of kitchen products",
    ],
    "fitness": [
        "actively consumes fitness content",
        "interested in workout routines",
        "potential buyer of sports equipment",
    ],
    "fashion": [
        "interested in fashion and style content",
        "frequently views clothing hauls",
        "potential buyer of apparel and accessories",
    ],
    "travel": [
        "interested in travel vlogs",
        "engages with destination content",
        "potential buyer of travel gear",
    ],
    "gaming": [
        "actively consumes gaming content",
        "interested in game reviews",
        "potential buyer of gaming peripherals",
    ],
    "beauty": [
        "interested in beauty tutorials",
        "engages with skincare and makeup content",
        "potential buyer of beauty products",
    ],
    "tech_review": [
        "interested in technology reviews",
        "frequently views gadget unboxings",
        "potential buyer of tech products",
    ],
    "home_decor": [
        "interested in home decoration content",
        "engages with interior design videos",
        "potential buyer of home goods",
    ],
    "pets": [
        "interested in pet care content",
        "frequently views animal videos",
        "potential buyer of pet supplies",
    ],
    "music": [
        "interested in music content",
        "engages with instrument and audio reviews",
        "potential buyer of musical equipment",
    ],
}

# Cross-domain purchase intent mapping
PURCHASE_INTENT_MAP = {
    "cooking": ["kitchen_appliance", "cookbook", "ingredient", "cooking_tool"],
    "fitness": ["sportswear", "supplement", "gym_equipment", "fitness_tracker"],
    "fashion": ["clothing", "accessories", "shoes", "jewelry"],
    "travel": ["luggage", "travel_gadget", "outdoor_gear", "travel_accessory"],
    "gaming": ["game", "gaming_peripheral", "console", "gaming_chair"],
    "beauty": ["skincare", "makeup", "haircare", "fragrance"],
    "tech_review": ["smartphone", "laptop", "smart_home", "wearable"],
    "home_decor": ["furniture", "lighting", "plants", "art"],
    "pets": ["pet_food", "pet_toy", "grooming_tool", "pet_accessory"],
    "music": ["instrument", "headphone", "audio_equipment", "music_software"],
}


class IntentAtom:
    """A single intent atom — the fundamental unit of AIR."""

    def __init__(
        self,
        atom_id: str,
        user_id: int,
        text: str,
        category: str,
        atom_type: str,  # "interest" | "purchase_intent"
        confidence: float,
        purchase_categories: Optional[list[str]] = None,
    ):
        self.atom_id = atom_id
        self.user_id = user_id
        self.text = text
        self.category = category
        self.atom_type = atom_type
        self.confidence = confidence
        self.purchase_categories = purchase_categories or []

    def to_dict(self) -> dict:
        return {
            "atom_id": self.atom_id,
            "user_id": self.user_id,
            "text": self.text,
            "category": self.category,
            "atom_type": self.atom_type,
            "confidence": self.confidence,
            "purchase_categories": self.purchase_categories,
        }


class IntentAtomizer:
    """
    Offline LLM-based intent atomizer.

    In production (per paper), a full LLM (e.g., 7B–70B) is used here.
    For this toy, we simulate the LLM with rule-based extraction.

    Paper formula (§3.1):
        A_u = LLM(B_u)  where B_u = {b1, b2, ..., bN} are user content behaviors
        Each atom a_i ∈ A_u is a (text, confidence, type) triple.
    """

    def __init__(self, use_mock_llm: bool = True):
        self.use_mock_llm = use_mock_llm

    def _extract_category_distribution(self, behaviors: list[dict]) -> dict:
        """Extract category engagement distribution from behaviors."""
        cat_counts = Counter()
        cat_duration = Counter()
        for b in behaviors:
            weight = {"view": 1, "like": 2, "share": 3, "comment": 2}.get(b["action"], 1)
            cat_counts[b["category"]] += weight
            cat_duration[b["category"]] += b.get("duration_seconds", 30)

        total = sum(cat_counts.values()) or 1
        return {
            cat: {
                "engagement_score": count / total,
                "avg_duration": cat_duration[cat] / (cat_counts[cat] or 1),
            }
            for cat, count in cat_counts.items()
        }

    def _mock_llm_atomize(self, user_id: int, behaviors: list[dict]) -> list[IntentAtom]:
        """
        Simulate LLM intent atomization.

        In the real paper, this runs offline on GPUs and produces
        rich natural-language intent descriptors.
        """
        cat_dist = self._extract_category_distribution(behaviors)
        atoms = []
        atom_idx = 0

        for cat, stats in cat_dist.items():
            confidence = min(1.0, stats["engagement_score"] * 3)

            if stats["engagement_score"] < 0.05:
                continue

            # Generate interest atoms
            templates = INTEREST_TEMPLATES.get(cat, [f"interested in {cat} content"])
            for tmpl in templates[:2]:  # top 2 templates per category
                atom_text = tmpl.format(action=random.choice(["viewing", "watching", "consuming"]) if "{action}" in tmpl else "")
                atoms.append(IntentAtom(
                    atom_id=f"u{user_id}_a{atom_idx}",
                    user_id=user_id,
                    text=atom_text,
                    category=cat,
                    atom_type="interest",
                    confidence=confidence,
                ))
                atom_idx += 1

            # Generate purchase intent atoms (cross-domain signal)
            purchase_cats = PURCHASE_INTENT_MAP.get(cat, [])
            if purchase_cats and stats["engagement_score"] > 0.1:
                intent_text = f"shows intent to purchase {', '.join(purchase_cats[:2])}"
                atoms.append(IntentAtom(
                    atom_id=f"u{user_id}_a{atom_idx}",
                    user_id=user_id,
                    text=intent_text,
                    category=cat,
                    atom_type="purchase_intent",
                    confidence=confidence * 0.8,
                    purchase_categories=purchase_cats,
                ))
                atom_idx += 1

        return atoms

    def atomize_user(self, user_id: int, behaviors: list[dict]) -> list[IntentAtom]:
        """Atomize a single user's behaviors into intent atoms."""
        if self.use_mock_llm:
            return self._mock_llm_atomize(user_id, behaviors)
        # TODO: integrate real LLM API here
        raise NotImplementedError("Real LLM integration not implemented in toy version.")

    def atomize_batch(
        self,
        behaviors_path: str,
        output_path: str,
    ) -> list[dict]:
        """Process all users' behaviors and save intent atoms."""
        with open(behaviors_path) as f:
            behaviors = json.load(f)

        # Group by user
        user_behaviors: dict[int, list] = {}
        for b in behaviors:
            uid = b["user_id"]
            user_behaviors.setdefault(uid, []).append(b)

        all_atoms = []
        for uid, user_b in user_behaviors.items():
            atoms = self.atomize_user(uid, user_b)
            all_atoms.extend([a.to_dict() for a in atoms])

        with open(output_path, "w") as f:
            json.dump(all_atoms, f, indent=2)

        print(f"Generated {len(all_atoms)} intent atoms for {len(user_behaviors)} users.")
        return all_atoms


# ── Dummy import for mock ──────────────────────────────────────────────────
import random  # noqa: E402 (moved here to avoid circular in future)


if __name__ == "__main__":
    import sys
    behaviors_path = sys.argv[1] if len(sys.argv) > 1 else "data/content_behaviors.json"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "data/atoms.json"

    atomizer = IntentAtomizer(use_mock_llm=True)
    atoms = atomizer.atomize_batch(behaviors_path, output_path)
    print(f"Sample atom: {atoms[0] if atoms else 'none'}")
