"""
GLiGuard: Schema-Conditioned Classification for LLM Safeguard
Reproduction of arXiv:2605.07982

Architecture:
  - Base: bidirectional encoder (~0.3B params, e.g. DeBERTa-v3-large or GLiNER2 backbone)
  - Schema injection: prepend a structured schema string specifying active moderation tasks
  - Multi-label classification head: outputs simultaneous predictions for all active tasks

Key insight from the paper:
  Instead of calling separate 7B–27B autoregressive models per safety task, GLiGuard
  frames ALL safety tasks as a single multi-label classification conditioned on a
  task schema, achieving 16× higher throughput and 17× lower latency.

Schema format (from paper):
  "[TASKS: prompt_safety, response_safety, harm_category, jailbreak_strategy]
   [INPUT] <conversation_text> [/INPUT]"

The model predicts labels for exactly the tasks listed in the schema.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict, Optional


# ---------------------------------------------------------------------------
# Task and label definitions (Section 3 of paper)
# ---------------------------------------------------------------------------

SAFETY_TASKS = [
    "prompt_safety",        # Is the prompt safe? (binary)
    "response_safety",      # Is the response safe? (binary)
    "refusal_detection",    # Did the model refuse? (binary)
]

HARM_CATEGORIES = [
    "sexual_content",
    "hate_speech",
    "self_harm",
    "violence",
    "harassment",
    "dangerous_instructions",
    "privacy_violation",
    "disinformation",
    "weapons",
    "cybercrime",
    "child_safety",
    "extremism",
    "financial_fraud",
    "other_harmful",
]  # 14 fine-grained harm categories (Section 2.2)

JAILBREAK_STRATEGIES = [
    "direct_request",
    "role_play",
    "prompt_injection",
    "few_shot_override",
    "system_prompt_override",
    "token_manipulation",
    "context_manipulation",
    "chain_of_thought_bypass",
    "obfuscation",
    "multi_turn_escalation",
    "hypothetical_framing",
]  # 11 jailbreak strategies (Section 2.3)

ALL_LABELS = SAFETY_TASKS + HARM_CATEGORIES + JAILBREAK_STRATEGIES
LABEL_TO_IDX = {label: idx for idx, label in enumerate(ALL_LABELS)}


# ---------------------------------------------------------------------------
# Schema Builder
# ---------------------------------------------------------------------------

class SchemaBuilder:
    """
    Builds the schema-conditioned input string from a list of active tasks.

    Paper: "Given an input schema specifying the selected moderation tasks,
    [GLiGuard] simultaneously evaluates prompt safety, response safety, fine-grained
    harm categories, and jailbreak strategies in a single forward pass."
    """

    @staticmethod
    def build(
        active_tasks: List[str],
        prompt: Optional[str] = None,
        response: Optional[str] = None,
    ) -> str:
        """
        Args:
            active_tasks: subset of ALL_LABELS to evaluate
            prompt: the user prompt (if evaluating prompt safety / jailbreak)
            response: the model response (if evaluating response safety / refusal)

        Returns:
            schema_conditioned_string ready for tokenizer
        """
        task_spec = ", ".join(active_tasks)
        parts = [f"[TASKS: {task_spec}]"]

        if prompt is not None:
            parts.append(f"[PROMPT] {prompt} [/PROMPT]")
        if response is not None:
            parts.append(f"[RESPONSE] {response} [/RESPONSE]")

        return " ".join(parts)


# ---------------------------------------------------------------------------
# GLiGuard Model
# ---------------------------------------------------------------------------

class GLiGuard(nn.Module):
    """
    Schema-conditioned bidirectional encoder for LLM content safety.

    Architecture (faithful to paper description):
      1. Tokenize schema-conditioned input
      2. Pass through ~0.3B bidirectional encoder (DeBERTa-v3-large or similar)
      3. Extract [CLS] token representation
      4. Multi-label classification head: sigmoid over active task logits

    The schema injection enables dynamic task selection without separate models.
    """

    def __init__(
        self,
        encoder_name: str = "microsoft/deberta-v3-large",
        num_labels: int = len(ALL_LABELS),
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout_rate)
        # Linear classification head: hidden_size → num_labels
        # Each label is predicted independently (multi-label)
        self.classifier = nn.Linear(hidden_size, num_labels)

        self.label_to_idx = LABEL_TO_IDX
        self.num_labels = num_labels

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        active_task_indices: Optional[List[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            token_type_ids: optional [batch, seq_len]
            active_task_indices: indices into ALL_LABELS for the schema-specified tasks

        Returns:
            dict with 'logits' [batch, num_labels] and 'probs' [batch, num_labels]
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Use [CLS] token representation
        # Paper: single forward pass, non-autoregressive
        cls_repr = outputs.last_hidden_state[:, 0, :]  # [batch, hidden]
        cls_repr = self.dropout(cls_repr)

        logits = self.classifier(cls_repr)  # [batch, num_labels]
        probs = torch.sigmoid(logits)        # multi-label: independent sigmoid

        result = {"logits": logits, "probs": probs}

        # If schema specifies active tasks, return only those predictions
        if active_task_indices is not None:
            result["active_logits"] = logits[:, active_task_indices]
            result["active_probs"] = probs[:, active_task_indices]

        return result

    def predict(
        self,
        text: str,
        active_tasks: List[str],
        tokenizer,
        threshold: float = 0.5,
        device: str = "cpu",
    ) -> Dict[str, float]:
        """
        High-level inference API mirroring the paper's multi-task single-pass design.

        Args:
            text: schema-conditioned input string (from SchemaBuilder)
            active_tasks: list of task names to evaluate
            tokenizer: HuggingFace tokenizer
            threshold: decision threshold for binary tasks
            device: computation device

        Returns:
            dict mapping task name → prediction (0/1 for binary, float prob for multi-label)
        """
        self.eval()
        encoding = tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,
        )
        encoding = {k: v.to(device) for k, v in encoding.items()}

        active_indices = [LABEL_TO_IDX[t] for t in active_tasks if t in LABEL_TO_IDX]

        with torch.no_grad():
            output = self.forward(
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"],
                active_task_indices=active_indices,
            )

        probs = output["active_probs"][0].cpu().tolist()
        predictions = {}
        for task, prob in zip(active_tasks, probs):
            predictions[task] = {
                "probability": prob,
                "label": int(prob >= threshold),
            }
        return predictions
