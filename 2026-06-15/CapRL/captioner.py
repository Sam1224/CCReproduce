"""
CapRL++ — Toy LVLM Captioner

Simulates the large vision-language model (LVLM) that generates captions
from images. In production: InternLM2.5-7B + CLIP-ViT-L or Qwen2.5-VL.

Paper (§2.2): The captioner's role:
    cap = LVLM(image)
    Trained with: L_RL = -R(cap) (maximize MCQ accuracy reward)

This toy simulates caption quality as a controllable parameter
to test the RL training loop dynamics.
"""
import json
import random
from typing import Optional


class ToyLVLMCaptioner:
    """
    Simulated LVLM captioner for CapRL++ toy reproduction.

    Generates captions of varying quality (controlled by 'skill_level').
    Trained via RL to maximize MCQ-based reward.

    In production (paper §2.2):
    - Architecture: InternLM2.5-7B + CLIP-ViT-L / Qwen2.5-VL variants
    - Training: RLVR with CapRL-Image-1M (1M image-MCQ pairs)
    - Result: 3B model ≥ Qwen2.5-VL-72B on Prism captioning
    """

    def __init__(self, skill_level: float = 0.5):
        """
        Args:
            skill_level: [0, 1] — 0=random captions, 1=perfect captions
                         Updated during RL training.
        """
        self.skill_level = skill_level
        self._step = 0

    def generate_caption(self, scene: dict) -> str:
        """
        Generate a caption for a scene.

        Quality controlled by skill_level (simulating RL training progress).
        """
        attrs = scene.get("ground_truth_attrs", {})
        full_desc = scene.get("full_description", "A scene.")

        if random.random() < self.skill_level:
            # High quality: include key attributes explicitly
            parts = []
            if "color" in attrs:
                parts.append(f"a {attrs['color']} colored {attrs.get('object', 'object')}")
            if "action" in attrs:
                parts.append(f"is {attrs['action']}")
            if "setting" in attrs:
                parts.append(f"in a {attrs['setting']}")

            if random.random() < self.skill_level:
                # Add extra detail for high skill
                parts.append(f"The scene is outdoors in natural light.")

            caption = " ".join(parts).capitalize() + "."
            if not caption.strip() or caption == ".":
                caption = full_desc
            return caption
        else:
            # Low quality: generic or partially correct
            templates = [
                "An animal is somewhere outdoors.",
                "Something is happening in a place.",
                f"A scene with a {attrs.get('object', 'thing')}.",
                "A colorful outdoor scene.",
                f"Something {attrs.get('action', 'visible')} in the environment.",
            ]
            return random.choice(templates)

    def update_skill(self, reward: float, lr: float = 0.05):
        """
        Simulate RL training: higher reward → higher skill level.
        In production: this is gradient descent on the LLM parameters.
        """
        self.skill_level = min(1.0, self.skill_level + lr * (reward - 0.5))
        self._step += 1

    def get_log_prob(self, caption: str, scene: dict) -> float:
        """
        Approximate log probability for REINFORCE.
        In production: computed from LLM token log probs.
        """
        attrs = scene.get("ground_truth_attrs", {})
        # Proxy: log_prob ~ how many attrs appear in caption
        caption_lower = caption.lower()
        n_present = sum(1 for v in attrs.values() if v.lower() in caption_lower)
        frac = n_present / (len(attrs) + 1e-8)
        # Smooth log prob
        import math
        return math.log(self.skill_level * frac + 0.01)


if __name__ == "__main__":
    # Test captioner at different skill levels
    scene = {
        "ground_truth_attrs": {
            "color": "red",
            "object": "dog",
            "action": "sitting",
            "setting": "park",
        },
        "full_description": "A red dog is sitting in a park.",
    }

    for skill in [0.1, 0.5, 0.9]:
        captioner = ToyLVLMCaptioner(skill_level=skill)
        cap = captioner.generate_caption(scene)
        print(f"Skill={skill:.1f}: {cap}")
