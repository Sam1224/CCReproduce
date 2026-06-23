"""
LLM-as-Enhancer backbone for Taiji.
Wraps a small LLM (e.g. Qwen/LLaMA-3B) for offline reasoning over user sequences.
The model is fine-tuned via SFT on reverse-engineered CoT data (Stage 1),
then aligned with POPO (Stage 2).
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional


class TaijiLLMEnhancer(nn.Module):
    """
    LLM-as-Enhancer: takes a user behavior sequence + context,
    outputs a reasoning trace (CoT) and a recommendation embedding.

    In the full Taiji system:
    - Offline: LLM reasons over user history → intent embedding
    - Online: lightweight model uses the intent embedding for ranking

    Here we implement the offline reasoning component.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        max_seq_len: int = 512,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.device = device
        self.max_seq_len = max_seq_len

        print(f"Loading tokenizer from {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading model from {model_name}...")
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
        ).to(device)

        # Intent projection head: maps LLM hidden state → intent embedding
        hidden_size = self.llm.config.hidden_size
        self.intent_projector = nn.Linear(hidden_size, 128).to(device)

    def format_user_sequence(self, user: dict) -> str:
        """Format user history into a prompt for CoT generation."""
        history_str = ""
        for h in user["history"][-10:]:  # last 10 interactions
            action = "purchased" if h["purchased"] else ("clicked" if h["clicked"] else "viewed")
            history_str += f"  - {action} item in {h['category']} ({h['price_tier']} price)\n"

        prompt = (
            f"User behavior analysis:\n"
            f"Recent interactions:\n{history_str}"
            f"Task: Reason about this user's purchase intent and recommend relevant ad categories.\n"
            f"Analysis:"
        )
        return prompt

    def forward(
        self,
        user_sequences: list[dict],
        labels: Optional[list[str]] = None,
    ) -> dict:
        """
        Forward pass for SFT training.

        Args:
            user_sequences: list of user dicts with 'history' field
            labels: CoT label strings (for SFT loss computation)

        Returns:
            dict with 'loss', 'intent_embeddings', 'logits'
        """
        prompts = [self.format_user_sequence(u) for u in user_sequences]

        if labels is not None:
            # SFT mode: format as prompt + completion
            full_texts = [p + " " + l for p, l in zip(prompts, labels)]
            inputs = self.tokenizer(
                full_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_seq_len,
            ).to(self.device)

            # Compute SFT loss only on completion tokens
            prompt_inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_seq_len,
            ).to(self.device)
            prompt_len = prompt_inputs["input_ids"].shape[1]

            labels_tensor = inputs["input_ids"].clone()
            labels_tensor[:, :prompt_len] = -100  # mask prompt tokens in loss

            outputs = self.llm(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=labels_tensor,
            )
            loss = outputs.loss

            # Extract intent embedding from last hidden state
            with torch.no_grad():
                hidden = outputs.hidden_states[-1] if outputs.hidden_states else None
        else:
            # Inference mode
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_seq_len,
            ).to(self.device)
            outputs = self.llm(**inputs, output_hidden_states=True)
            loss = None
            hidden = outputs.hidden_states[-1]

        # Intent embedding: mean-pool last hidden state
        intent_embeddings = None
        if hidden is not None:
            pooled = hidden.mean(dim=1)  # [B, hidden_size]
            intent_embeddings = self.intent_projector(pooled)  # [B, 128]

        return {
            "loss": loss,
            "intent_embeddings": intent_embeddings,
            "logits": outputs.logits if hasattr(outputs, "logits") else None,
        }

    def generate_cot(self, user: dict, max_new_tokens: int = 200) -> str:
        """Generate CoT reasoning for a single user (inference mode)."""
        prompt = self.format_user_sequence(user)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            generated = self.llm.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        decoded = self.tokenizer.decode(generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return decoded


class TaijiOnlineRanker(nn.Module):
    """
    Lightweight online ranking model that consumes intent embeddings
    produced by the offline TaijiLLMEnhancer.

    In production Taiji:
    - Offline: LLMEnhancer generates intent_embedding per user
    - Online: this lightweight MLP predicts CTR/CVR using intent_embedding + ID features
    """

    def __init__(self, intent_dim: int = 128, id_dim: int = 64, n_tasks: int = 2):
        super().__init__()
        self.intent_proj = nn.Linear(intent_dim, 64)
        self.id_proj = nn.Linear(id_dim, 64)

        self.shared_mlp = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        # Multi-task heads: CTR and CVR
        self.task_heads = nn.ModuleList([
            nn.Linear(128, 1) for _ in range(n_tasks)
        ])

    def forward(
        self,
        intent_embeddings: torch.Tensor,   # [B, 128] from offline LLM
        item_id_embeddings: torch.Tensor,  # [B, 64] from ID lookup table
    ) -> dict:
        intent_feat = self.intent_proj(intent_embeddings)     # [B, 64]
        id_feat = self.id_proj(item_id_embeddings)            # [B, 64]

        combined = torch.cat([intent_feat, id_feat], dim=-1)  # [B, 128]
        shared = self.shared_mlp(combined)                     # [B, 128]

        outputs = {}
        task_names = ["ctr", "cvr"]
        for i, head in enumerate(self.task_heads):
            outputs[task_names[i]] = torch.sigmoid(head(shared)).squeeze(-1)  # [B]
        return outputs
