"""
Unified model interface for EVADE evaluation.
Supports GPT-4o, Claude, Qwen-VL, and other LLM/VLM backends.
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)


class ModelInterface(ABC):
    """Abstract base class for LLM/VLM model backends."""

    @abstractmethod
    def predict(
        self,
        prompt: str,
        system: Optional[str] = None,
        image_path: Optional[str] = None,
        max_tokens: int = 16,
        temperature: float = 0.0,
    ) -> str:
        raise NotImplementedError


class DummyModel(ModelInterface):
    """Dummy model for testing without API access."""

    def predict(self, prompt: str, system=None, image_path=None, max_tokens=16, temperature=0.0) -> str:
        # Simple keyword-based heuristic for toy evaluation
        keywords_evasive = ["保证", "一定", "快速", "立竿见影", "无副作用", "纯天然", "100%"]
        if any(kw in prompt for kw in keywords_evasive):
            return "违规"
        return "合规"


class GPT4Model(ModelInterface):
    """OpenAI GPT-4o/GPT-4V interface."""

    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required: pip install openai")

        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.model = model

    def predict(
        self,
        prompt: str,
        system: Optional[str] = None,
        image_path: Optional[str] = None,
        max_tokens: int = 16,
        temperature: float = 0.0,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})

        if image_path and self.model in ("gpt-4o", "gpt-4-vision-preview"):
            import base64
            with open(image_path, "rb") as f:
                img_data = base64.b64encode(f.read()).decode()
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_data}"}},
            ]
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt})

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()


class ClaudeModel(ModelInterface):
    """Anthropic Claude interface."""

    def __init__(self, model: str = "claude-opus-4-7", api_key: Optional[str] = None):
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package required: pip install anthropic")

        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model

    def predict(
        self,
        prompt: str,
        system: Optional[str] = None,
        image_path: Optional[str] = None,
        max_tokens: int = 16,
        temperature: float = 0.0,
    ) -> str:
        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
        }
        if system:
            kwargs["system"] = system

        if image_path:
            import base64
            with open(image_path, "rb") as f:
                img_data = base64.b64encode(f.read()).decode()
            content = [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_data}},
                {"type": "text", "text": prompt},
            ]
        else:
            content = prompt

        resp = self.client.messages.create(
            messages=[{"role": "user", "content": content}],
            **kwargs,
        )
        return resp.content[0].text.strip()


def get_model(model_name: str) -> ModelInterface:
    """Factory function to create model by name."""
    if model_name == "dummy":
        return DummyModel()
    elif model_name.startswith("gpt"):
        return GPT4Model(model=model_name)
    elif model_name.startswith("claude"):
        return ClaudeModel(model=model_name)
    else:
        raise ValueError(f"Unknown model: {model_name}. Supported: dummy, gpt-4o, claude-*")
