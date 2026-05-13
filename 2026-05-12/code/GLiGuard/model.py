"""
GLiGuard model implementation.

Architecture (Figure 1 in paper):

  1. Schema Prefix Construction:
     All active task schemas are serialized as structured token prefixes:
       [TASK: name: desc] [LABEL: name: desc] ... [SEP] <content> [SEP]

  2. Bidirectional Encoder:
     The concatenated schema+content sequence is fed into a bidirectional
     transformer encoder (DeBERTa-v3 in this reproduction).
     Bidirectional attention enables labels to attend over content and vice-versa.

  3. Per-Label Classification:
     The paper uses the hidden state of each [LABEL:...] token span as the
     representation for that label's binary classifier.
     We approximate this with token-span mean pooling over label tokens + a
     linear classification head for each label slot.

  4. Schema-Conditioned Multi-task Output:
     All label decisions for all active schemas are produced in ONE forward pass.

Key formula (Section 3.2 in paper):
  h = Encoder([schema_prefix; SEP; content; SEP])         # bidirectional
  For label i with token span [s_i, e_i]:
    z_i = mean(h[s_i:e_i])                                # label representation
    p_i = sigmoid(W_i z_i + b_i)                          # binary classifier
    loss_i = BCE(p_i, y_i)                                # per-label loss
  total_loss = sum(loss_i for all active labels)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from dataclasses import dataclass, field
from typing import Optional

from schemas import (
    ALL_SCHEMAS,
    PROMPT_SAFETY_SCHEMA,
    RESPONSE_SAFETY_SCHEMA,
    REFUSAL_SCHEMA,
    HARM_CATEGORY_SCHEMA,
    JAILBREAK_SCHEMA,
    build_schema_prefix,
    get_all_label_names,
)


@dataclass
class GLiGuardConfig:
    encoder_name: str = "microsoft/deberta-v3-base"
    # Map task_name → list of label names
    task_label_map: dict = field(default_factory=lambda: {
        s["task_name"]: get_all_label_names(s) for s in ALL_SCHEMAS
    })
    max_length: int = 512
    dropout: float = 0.1
    # Threshold for converting sigmoid output to binary prediction
    threshold: float = 0.5


class GLiGuardModel(nn.Module):
    """
    GLiGuard: Schema-Conditioned Bidirectional Encoder for LLM Safety Classification.

    This reproduction implements the core architecture:
    - Schema prefix encoding (task + label descriptions as input tokens)
    - Bidirectional encoder (DeBERTa-v3-base as backbone)
    - Per-label binary classification via span mean pooling

    The model supports multi-task inference: given a content string, it can
    classify across all task schemas in one forward pass by concatenating
    all schema prefixes into the input.
    """

    def __init__(self, config: GLiGuardConfig):
        super().__init__()
        self.config = config
        self.encoder = AutoModel.from_pretrained(config.encoder_name)
        self.tokenizer = AutoTokenizer.from_pretrained(config.encoder_name)

        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(config.dropout)

        # One linear classifier per label per task
        # task_name → {label_name → Linear}
        self.classifiers: nn.ModuleDict = nn.ModuleDict()
        for task_name, labels in config.task_label_map.items():
            task_classifiers = nn.ModuleDict({
                label: nn.Linear(hidden_size, 1)
                for label in labels
            })
            self.classifiers[task_name] = task_classifiers

    def encode_with_schema(
        self,
        content: str,
        active_schemas: list[dict],
        device: torch.device,
    ) -> tuple[torch.Tensor, dict[str, dict[str, tuple[int, int]]]]:
        """
        Build input sequence: [schema_prefix] [SEP] [content] [SEP]
        Returns:
          - last_hidden_state: (1, seq_len, hidden_size)
          - label_spans: task_name → label_name → (start_idx, end_idx) in token space
        """
        # Build full prefix string
        schema_strings = [build_schema_prefix(s) for s in active_schemas]
        full_prefix = " ".join(schema_strings)
        full_text = full_prefix + " " + content

        # Tokenize
        encoding = self.tokenizer(
            full_text,
            return_tensors="pt",
            max_length=self.config.max_length,
            truncation=True,
            padding="max_length",
        ).to(device)

        # Forward pass through encoder
        outputs = self.encoder(**encoding)
        hidden = outputs.last_hidden_state  # (1, seq_len, hidden_size)

        # Locate label token spans in input_ids
        # We use a simplified heuristic: the [LABEL: name:] tokens are found by
        # tokenizing each label name independently and searching in the full sequence.
        # In the real paper, special tokens are used to mark label boundaries precisely.
        label_spans: dict[str, dict[str, tuple[int, int]]] = {}
        input_ids = encoding["input_ids"][0]

        for schema in active_schemas:
            task_name = schema["task_name"]
            label_spans[task_name] = {}
            for label in schema["labels"]:
                label_name = label["name"]
                # Tokenize label name to find its position
                label_ids = self.tokenizer(
                    label_name,
                    add_special_tokens=False,
                )["input_ids"]
                span = self._find_span(input_ids.tolist(), label_ids)
                label_spans[task_name][label_name] = span

        return hidden, label_spans

    @staticmethod
    def _find_span(seq: list[int], subseq: list[int]) -> tuple[int, int]:
        """Find first occurrence of subseq in seq; return (start, end) or (0, 1) if not found."""
        n, m = len(seq), len(subseq)
        for i in range(n - m + 1):
            if seq[i:i + m] == subseq:
                return (i, i + m)
        # Fallback: use [CLS] representation when label not found in truncated sequence
        return (0, 1)

    def forward(
        self,
        content: str,
        active_schemas: list[dict],
        device: Optional[torch.device] = None,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """
        Returns logits dict: task_name → {label_name → scalar logit tensor}

        All active schemas are evaluated in ONE forward pass.
        """
        if device is None:
            device = next(self.parameters()).device

        hidden, label_spans = self.encode_with_schema(content, active_schemas, device)
        hidden = self.dropout(hidden)  # (1, seq_len, hidden_size)

        logits: dict[str, dict[str, torch.Tensor]] = {}

        for schema in active_schemas:
            task_name = schema["task_name"]
            logits[task_name] = {}
            for label in schema["labels"]:
                label_name = label["name"]
                start, end = label_spans[task_name][label_name]
                # Mean pool over label token span to get label representation
                # z_i = mean(h[s_i:e_i])  — eq. (3) in paper
                label_repr = hidden[0, start:end, :].mean(dim=0)  # (hidden_size,)
                # Binary classification: p_i = sigmoid(W_i z_i + b_i)
                logit = self.classifiers[task_name][label_name](label_repr)  # (1,)
                logits[task_name][label_name] = logit.squeeze(-1)

        return logits

    def predict(
        self,
        content: str,
        active_schemas: list[dict],
        threshold: Optional[float] = None,
        device: Optional[torch.device] = None,
    ) -> dict[str, dict[str, int]]:
        """
        Run inference and return binary predictions.
        Returns: task_name → {label_name → 0/1}
        """
        if threshold is None:
            threshold = self.config.threshold
        with torch.no_grad():
            logits = self.forward(content, active_schemas, device)
        predictions: dict[str, dict[str, int]] = {}
        for task_name, label_logits in logits.items():
            predictions[task_name] = {}
            for label_name, logit in label_logits.items():
                prob = torch.sigmoid(logit).item()
                predictions[task_name][label_name] = int(prob >= threshold)
        return predictions


class GLiGuardLoss(nn.Module):
    """
    Multi-label binary cross-entropy loss aggregated over all active task schemas.

    total_loss = (1/N) * sum_{task} sum_{label_i in task} BCE(p_i, y_i)
    where N = total number of active label slots.
    """

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(
        self,
        logits: dict[str, dict[str, torch.Tensor]],
        targets: dict[str, dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        losses = []
        for task_name, label_logits in logits.items():
            if task_name not in targets:
                continue
            for label_name, logit in label_logits.items():
                if label_name not in targets[task_name]:
                    continue
                target = targets[task_name][label_name].float()
                losses.append(self.bce(logit.unsqueeze(0), target.unsqueeze(0)))
        if not losses:
            return torch.tensor(0.0, requires_grad=True)
        return torch.stack(losses).mean()


def build_model(encoder_name: str = "microsoft/deberta-v3-base") -> GLiGuardModel:
    config = GLiGuardConfig(encoder_name=encoder_name)
    return GLiGuardModel(config)


if __name__ == "__main__":
    # Smoke test
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_model("microsoft/deberta-v3-base").to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Test single forward pass
    active = [PROMPT_SAFETY_SCHEMA, HARM_CATEGORY_SCHEMA, JAILBREAK_SCHEMA]
    test_prompt = "How do I make a bomb?"
    preds = model.predict(test_prompt, active, device=device)

    print("\nPredictions for unsafe prompt:")
    for task, labels in preds.items():
        positives = [l for l, v in labels.items() if v == 1]
        print(f"  {task}: {positives if positives else ['(none)']}")
