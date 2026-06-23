"""
LLM Offline Reasoner for AIR.

Takes Atomic Intent Units (AIUs) as input — not raw behavior sequences —
and reasons about the user's cross-domain purchase intent.

Key differences from standard LLM-for-recommendation:
1. Input is compressed AIUs, not noisy raw history (noise is already filtered)
2. Focus is cross-domain: content engagement → e-commerce purchase prediction
3. Output is a cross-domain intent embedding (D=256), not item IDs
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional


SYSTEM_PROMPT = (
    "You are an expert cross-domain recommendation analyst. "
    "Given a user's compressed behavioral signals from content consumption, "
    "reason about their likely e-commerce purchase intent."
)


def format_aiu_prompt(aiu_text: str, user_id: Optional[int] = None) -> str:
    """Format AIU text into a reasoning prompt for the LLM."""
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"User Atomic Intent Summary:\n{aiu_text}\n\n"
        "Based on the above intent signals, analyze:\n"
        "1. What e-commerce product categories match this user's content interests?\n"
        "2. What price tier are they likely to purchase in?\n"
        "3. What is their cross-domain conversion likelihood (low/medium/high)?\n"
        "Cross-domain purchase intent reasoning:"
    )


class AIRLLMReasoner(nn.Module):
    """
    LLM-based cross-domain intent reasoner.
    Runs OFFLINE — generates intent embeddings for each user once per day.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        intent_dim: int = 256,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = device
        self.intent_dim = intent_dim

        print(f"Loading AIR LLM reasoner from {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
        ).to(device)

        # Project LLM hidden states to cross-domain intent embedding
        hidden_size = self.llm.config.hidden_size
        self.cross_domain_projector = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Linear(512, intent_dim),
            nn.LayerNorm(intent_dim),
        ).to(device)

    def embed_aiu(self, aiu_texts: list[str]) -> torch.Tensor:
        """
        Convert AIU texts to cross-domain intent embeddings.
        Returns: [B, intent_dim]
        """
        prompts = [format_aiu_prompt(t) for t in aiu_texts]
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.llm(**inputs, output_hidden_states=True)

        # Last hidden state, mean-pooled
        hidden = outputs.hidden_states[-1]  # [B, T, H]
        pooled = hidden.mean(dim=1)         # [B, H]

        return self.cross_domain_projector(pooled.float())  # [B, intent_dim]

    def forward(
        self,
        aiu_texts: list[str],
        labels: Optional[list[str]] = None,
    ) -> dict:
        """
        Forward pass for SFT training on cross-domain reasoning.

        Args:
            aiu_texts: AIU strings (already extracted, clean intent signals)
            labels: target reasoning text (for SFT loss)

        Returns:
            dict with 'loss', 'intent_embeddings'
        """
        prompts = [format_aiu_prompt(t) for t in aiu_texts]

        if labels is not None:
            full_texts = [p + " " + l for p, l in zip(prompts, labels)]
            inputs = self.tokenizer(
                full_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            # Mask prompt tokens from loss
            prompt_inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)
            prompt_len = prompt_inputs["input_ids"].shape[1]

            label_ids = inputs["input_ids"].clone()
            label_ids[:, :prompt_len] = -100

            outputs = self.llm(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=label_ids,
                output_hidden_states=True,
            )
            loss = outputs.loss
            hidden = outputs.hidden_states[-1]
        else:
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)
            outputs = self.llm(**inputs, output_hidden_states=True)
            loss = None
            hidden = outputs.hidden_states[-1]

        pooled = hidden.mean(dim=1)
        intent_embeddings = self.cross_domain_projector(pooled.float())  # [B, intent_dim]

        return {
            "loss": loss,
            "intent_embeddings": intent_embeddings,
        }

    def generate_reasoning(self, aiu_text: str, max_new_tokens: int = 200) -> str:
        """Generate cross-domain intent reasoning for a single user (inference)."""
        prompt = format_aiu_prompt(aiu_text)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            generated = self.llm.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(
            generated[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
