from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class HFTextLM:
    """A minimal HuggingFace causal LM adapter.

    IECD[2] is proposed for VLMs. This adapter exists to make the repo runnable on CPU
    with a small model (e.g. distilgpt2) while keeping the IECD math identical.

    For a real VLM (e.g. LLaVA), you would replace this with an adapter that:
    1) formats multi-modal inputs (prompt + image) as model inputs;
    2) returns next-token logits for both instruction/evidence prompts.
    """

    model_name: str = "distilgpt2"
    device: str = "cpu"

    def __post_init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def next_token_logits(self, prompt: str, input_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        if input_ids is None:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

        out = self.model(input_ids=input_ids)
        return out.logits[:, -1, :]
