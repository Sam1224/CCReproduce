"""
VLM Wrapper for RuleSafe-VL Evaluation
=========================================
Thin wrapper supporting multiple HuggingFace VLMs via a unified interface.
Supported backends:
  - "random"     : random baseline (for pipeline testing without a real model)
  - "llava-1.6"  : LLaVA-1.6 (llava-hf/llava-v1.6-mistral-7b-hf)
  - "internvl2"  : InternVL2 (OpenGVLab/InternVL2-8B)
  - "openai"     : OpenAI API (gpt-4o, gpt-5.2, etc.) via environment variable OPENAI_API_KEY
"""

import os
import json
import random
from typing import Optional


class RandomBaselineWrapper:
    """Trivial random baseline for pipeline testing."""
    OUTCOMES = ["ALLOW", "RESTRICT", "REMOVE"]

    def generate(self, prompt: str, image_path: Optional[str] = None) -> str:
        all_rule_ids = ["F1_R01", "F1_R02", "F1_R03", "F2_R01", "F2_R02", "F3_R01"]
        n_applicable = random.randint(1, 3)
        applicable = random.sample(all_rule_ids, min(n_applicable, len(all_rule_ids)))
        n_violated = random.randint(0, len(applicable))
        violated = random.sample(applicable, n_violated)
        outcome = random.choice(self.OUTCOMES)
        response = {
            "applicable_rules": applicable,
            "violated_rules": violated,
            "moderation_outcome": outcome,
            "reasoning_chain": [],
            "confidence": random.uniform(0.3, 0.9),
        }
        return json.dumps(response)


class LLaVAWrapper:
    """LLaVA-1.6 wrapper using HuggingFace transformers."""

    def __init__(self, model_name: str = "llava-hf/llava-v1.6-mistral-7b-hf"):
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
        import torch
        self.processor = LlavaNextProcessor.from_pretrained(model_name)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )
        self.model.eval()

    def generate(self, prompt: str, image_path: Optional[str] = None,
                 max_new_tokens: int = 512) -> str:
        from PIL import Image
        import torch

        messages = [
            {"role": "system", "content": "You are a content moderation assistant. Always respond in valid JSON."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
            ]}
        ]
        if image_path:
            messages[-1]["content"].insert(0, {"type": "image"})

        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        images = [Image.open(image_path)] if image_path else None
        inputs = self.processor(text=text, images=images, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

        generated = self.processor.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return generated


class OpenAIWrapper:
    """OpenAI API wrapper (GPT-4o, GPT-5.2, etc.)."""

    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None):
        import openai
        self.client = openai.OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])
        self.model = model

    def generate(self, prompt: str, image_path: Optional[str] = None,
                 max_tokens: int = 1024) -> str:
        import base64
        messages = [{"role": "system", "content": "You are a content moderation assistant. Always respond in valid JSON."}]
        user_content = [{"type": "text", "text": prompt}]
        if image_path:
            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()
            user_content.insert(0, {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
            })
        messages.append({"role": "user", "content": user_content})
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content


def get_model_wrapper(model_name: str):
    """Factory function to get the appropriate VLM wrapper."""
    if model_name == "random":
        return RandomBaselineWrapper()
    elif model_name.startswith("llava"):
        return LLaVAWrapper()
    elif model_name.startswith("gpt") or model_name.startswith("openai"):
        return OpenAIWrapper(model=model_name)
    else:
        raise ValueError(f"Unknown model: {model_name}. Supported: random, llava-1.6, gpt-4o, ...")
