"""
Stage III: Latent Knowledge Discovery.

From paper §3.3:
    "Stage III employs a tripartite dialectical discussion to unearth sophisticated
     'gray-area' violations. The system mines hard adversarial cases by synthesizing
     samples that sit at the decision boundary, forcing the model to develop more
     nuanced representations of policy violations."

Hard sample mining strategy (paper, inferred):
    1. Identify gray-area samples (where Prosecutor ≈ Defender scores)
    2. For each gray-area sample, generate semantic variants via:
       a. Lexical substitution (e.g., replace direct violation keywords with euphemisms)
       b. Structural paraphrase (reorder clauses to obscure violation signal)
    3. Assign confident labels via the Umpire's final verdict
    4. Add hard samples to training set D_final
"""

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple


# Violation euphemism pairs — simulates LLM-generated paraphrases
EUPHEMISM_PAIRS = {
    "left behind": "not keeping up",
    "fall behind": "miss out",
    "fail": "struggle",
    "ugly": "needs improvement",
    "old": "mature",
    "fading": "declining",
    "guaranteed": "clinically studied",
    "miracle": "advanced formula",
    "doctor-approved": "recommended by professionals",
    "fat": "carrying extra weight",
    "wrinkle": "fine lines",
}


class LatentKnowledgeDiscovery:
    """
    Mines hard adversarial samples from gray-area cases.

    In production (paper):
        - Uses an LLM to generate nuanced paraphrases
        - Uses retrieval to find similar historical cases
        - Applies a curriculum: easy → medium → hard samples

    In this toy:
        - Applies lexical substitution to create "gray-area" variants
        - Assigns label from the original rectified label
    """

    def __init__(self, model=None, n_variants: int = 2):
        self.model = model
        self.n_variants = n_variants

    def _apply_euphemisms(self, text: str) -> str:
        result = text
        for direct, euphemism in EUPHEMISM_PAIRS.items():
            if direct in result.lower():
                result = result.lower().replace(direct, euphemism)
                break
        return result

    def _structural_paraphrase(self, text: str) -> str:
        sentences = text.split(". ")
        if len(sentences) > 1:
            random.shuffle(sentences)
            return ". ".join(sentences)
        return text + " Learn more today."

    def _generate_variant(self, text: str, variant_type: int) -> str:
        if variant_type == 0:
            return self._apply_euphemisms(text)
        else:
            return self._structural_paraphrase(text)

    def mine_hard_samples(
        self,
        gray_area_cases: List[Tuple[str, str]],
    ) -> List[dict]:
        """
        Generate hard adversarial training samples from gray-area cases.

        Args:
            gray_area_cases: list of (text, rectified_label) tuples

        Returns:
            list of {text, new_label, source, split} dicts
        """
        hard_samples = []
        for text, label in gray_area_cases:
            for i in range(self.n_variants):
                variant = self._generate_variant(text, i)
                hard_samples.append({
                    "text": variant,
                    "new_label": label,
                    "old_label": label,
                    "source": "stage3_hard_mining",
                    "split": "train",
                })
        return hard_samples

    def discover_latent_violations(
        self,
        samples: List[dict],
        threshold: float = 0.3,
    ) -> List[dict]:
        """
        Identify potential hidden violations via tripartite discussion.

        Paper §3.3: "Latent Knowledge Discovery unearths sophisticated violations
        that do not trigger surface-level keyword matching."

        In toy: uses model confidence as a proxy for latent violation signal.
        """
        if self.model is None:
            return []

        latent_violations = []
        self.model.eval()
        import torch
        with torch.no_grad():
            for sample in samples:
                label, conf = self.model.predict(sample["text"])
                if label == "violation" and conf < threshold:
                    latent_violations.append({**sample, "latent": True})
        return latent_violations
