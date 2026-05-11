"""
Valley3 Controllable Reasoning Modes.

Paper Section 4.3: Valley3 supports one non-thinking mode and three
distinct levels of thinking to cover different latency requirements:

  Mode 0 - Non-thinking:
    Fast response, no explicit CoT. Used for real-time content moderation,
    quick product attribute lookup.

  Mode 1 - Light thinking:
    Brief chain-of-thought (1-3 steps). Used for product compliance quick checks,
    basic attribute extraction with simple justification.

  Mode 2 - Medium thinking:
    Moderate CoT (3-7 steps). Used for multimodal product analysis,
    short-video content quality assessment.

  Mode 3 - Heavy thinking:
    Deep CoT (7+ steps). Used for complex compliance decisions,
    competitive product analysis, multi-product research tasks.

Implementation: Prepend mode-specific system prompt tokens before generation.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import torch


@dataclass
class ReasoningModeConfig:
    """Configuration for a specific reasoning mode."""
    name: str
    max_thinking_tokens: int
    system_prefix: str
    temperature: float
    top_p: float


REASONING_MODES: Dict[str, ReasoningModeConfig] = {
    "non_thinking": ReasoningModeConfig(
        name="non_thinking",
        max_thinking_tokens=0,
        system_prefix="<|non_thinking_start|>请直接给出答案，无需解释。<|non_thinking_end|>",
        temperature=0.0,  # Greedy for consistency
        top_p=1.0,
    ),
    "light": ReasoningModeConfig(
        name="light",
        max_thinking_tokens=128,
        system_prefix="<|thinking_start|>请简要分析后给出答案（2-3步）。<|thinking_end|>",
        temperature=0.3,
        top_p=0.9,
    ),
    "medium": ReasoningModeConfig(
        name="medium",
        max_thinking_tokens=512,
        system_prefix="<|thinking_start|>请认真分析，给出详细推理过程。<|thinking_end|>",
        temperature=0.5,
        top_p=0.9,
    ),
    "heavy": ReasoningModeConfig(
        name="heavy",
        max_thinking_tokens=2048,
        system_prefix=(
            "<|thinking_start|>请进行深度分析。考虑多个角度，"
            "结合视觉信息和文本信息，给出完整的推理链。<|thinking_end|>"
        ),
        temperature=0.7,
        top_p=0.95,
    ),
}


class ReasoningModeController:
    """
    Controls Valley3's reasoning mode at inference time.

    Usage:
        controller = ReasoningModeController()
        prompt = controller.apply_mode(
            user_prompt="这个商品是否存在虚假宣传？",
            mode="heavy"
        )
    """

    def __init__(self):
        self.modes = REASONING_MODES

    def apply_mode(self, user_prompt: str, mode: str = "non_thinking") -> str:
        """
        Apply reasoning mode prefix to user prompt.

        Args:
            user_prompt: The actual user instruction
            mode: One of "non_thinking", "light", "medium", "heavy"
        Returns:
            Full prompt with mode-specific prefix
        """
        if mode not in self.modes:
            raise ValueError(f"Unknown mode '{mode}'. Choose from: {list(self.modes.keys())}")

        config = self.modes[mode]
        return f"{config.system_prefix}\n\n{user_prompt}"

    def get_generation_config(self, mode: str) -> Dict[str, Any]:
        """Get generation hyperparameters for a specific mode."""
        config = self.modes[mode]
        return {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_new_tokens": config.max_thinking_tokens + 512,  # thinking + response
        }

    def select_mode_for_task(self, task_type: str) -> str:
        """
        Automatically select reasoning mode based on e-commerce task type.
        Valley3's practical deployment heuristic.
        """
        task_mode_map = {
            # Real-time tasks → non_thinking
            "realtime_moderation": "non_thinking",
            "quick_attribute_lookup": "non_thinking",

            # Standard tasks → light
            "product_tagging": "light",
            "spam_detection": "light",
            "basic_compliance_check": "light",

            # Analysis tasks → medium
            "product_quality_assessment": "medium",
            "short_video_analysis": "medium",
            "influencer_content_review": "medium",

            # Complex tasks → heavy
            "deep_compliance_review": "heavy",
            "competitive_product_research": "heavy",
            "cross_modal_violation_detection": "heavy",
        }
        return task_mode_map.get(task_type, "medium")


def demonstrate_modes():
    """Demonstrate all reasoning modes on an e-commerce task."""
    controller = ReasoningModeController()

    task = "请分析这个商品短视频，判断是否存在虚假宣传：产品声称'7天瘦10斤，无需运动'"

    print("=" * 60)
    print("Valley3 Reasoning Mode Demonstration")
    print("Task:", task[:50] + "...")
    print("=" * 60)

    for mode_name in ["non_thinking", "light", "medium", "heavy"]:
        full_prompt = controller.apply_mode(task, mode=mode_name)
        gen_config = controller.get_generation_config(mode_name)

        print(f"\n[Mode: {mode_name.upper()}]")
        print(f"  Prompt prefix: {full_prompt[:80]}...")
        print(f"  Max thinking tokens: {REASONING_MODES[mode_name].max_thinking_tokens}")
        print(f"  Temperature: {gen_config['temperature']}")

        # Simulate response (in real deployment, pass to Valley3Model.generate())
        if mode_name == "non_thinking":
            print(f"  → Response: [VIOLATION] 违规 - 不实减肥宣传")
        elif mode_name == "light":
            print(f"  → Response: 分析：声称7天瘦10斤不符合医学常识 → [VIOLATION]")
        elif mode_name == "medium":
            print(f"  → Response: 多角度分析：视觉内容与文字声明对比 → 缺乏科学依据 → [VIOLATION]")
        else:
            print(f"  → Response: 深度分析：(1)医学角度 (2)法规角度 (3)视觉证据 → [SERIOUS VIOLATION]")


if __name__ == "__main__":
    demonstrate_modes()
