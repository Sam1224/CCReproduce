from __future__ import annotations

from dataclasses import dataclass

import torch

from dataset import TEMPLATE_POOL, LogGenExample, tokenize
from model import ReLogClassifier


@dataclass(frozen=True)
class GeneratedLog:
    level: int
    template: str


def _score_log_sufficiency(*, log_template: str, trace: str) -> float:
    """Proxy for the paper's downstream-task-based log evaluation.

    ReLog evaluates logs by their utility for downstream debugging (defect
    localization/repair). In this toy reproduction, we approximate usefulness by
    checking whether the template covers key runtime fields.
    """

    trace_tokens = set(tokenize(trace))
    covered = 0
    for field in ("x=", "y=", "args=", "result=", "error="):
        if field in trace and field.strip("=") in log_template:
            covered += 1
        if any(t.startswith(field) for t in trace_tokens) and field.strip("=") in log_template:
            covered += 1
    return float(covered)


def generate_initial_log(
    model: ReLogClassifier,
    *,
    example: LogGenExample,
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> GeneratedLog:
    model.eval()
    with torch.no_grad():
        out = model(token_ids.unsqueeze(0), attention_mask.unsqueeze(0))
        level = int(out.level_logits.argmax(dim=-1).item())
        template_id = int(out.template_logits.argmax(dim=-1).item())
    return GeneratedLog(level=level, template=TEMPLATE_POOL[template_id])


def refine_log(
    model: ReLogClassifier,
    *,
    example: LogGenExample,
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    top_k: int = 3,
) -> GeneratedLog:
    """Execution-aware refinement loop (toy).

    We produce top-k templates from the model and select the one with the best
    sufficiency score using runtime trace.
    """

    model.eval()
    with torch.no_grad():
        out = model(token_ids.unsqueeze(0), attention_mask.unsqueeze(0))
        template_scores = out.template_logits.squeeze(0)
        top_values, top_indices = torch.topk(template_scores, k=min(top_k, template_scores.numel()))

    best_level = int(out.level_logits.argmax(dim=-1).item())
    best_template = TEMPLATE_POOL[int(top_indices[0].item())]
    best_score = -1.0

    for idx in top_indices.tolist():
        template = TEMPLATE_POOL[int(idx)]
        score = _score_log_sufficiency(log_template=template, trace=example.runtime_trace)
        if score > best_score:
            best_score = score
            best_template = template

    return GeneratedLog(level=best_level, template=best_template)
