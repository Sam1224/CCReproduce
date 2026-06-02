"""
LLM Teacher for CS-VAR knowledge distillation.

In the paper, an LLM reasons over:
  (a) Current session behavioral summary
  (b) Retrieved cross-session evidence (K historical sessions)
→ Produces structured risk assessment with local-to-global pattern recognition.

This module provides:
  1. A prompt template for the LLM teacher
  2. A lightweight surrogate teacher (MLP) for reproducibility without LLM access
  3. KD loss for distilling teacher outputs to the student model

Paper §3.2: "LLM reasons over retrieved cross-session behavioral evidence and
transfers its local-to-global insights to the small model"

Pseudocode for real LLM teacher call:
    prompt = build_risk_prompt(current_session_summary, retrieved_sessions)
    response = llm.generate(prompt)
    risk_logits = parse_structured_output(response)  # extract risk level probabilities
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Prompt template (used with real LLM API)
# ---------------------------------------------------------------------------
RISK_ASSESSMENT_PROMPT = """
You are a live streaming risk assessment expert. Analyze the following session.

## Current Session Summary
{current_session}

## Retrieved Similar Sessions (Historical Evidence)
{retrieved_sessions}

## Task
Based on the pattern of behavior across sessions, assess the risk level:
- 0: Low risk (no suspicious patterns)
- 1: Medium risk (some concerning patterns)
- 2: High risk (clear malicious patterns, e.g., recurring scam scripts, coordinated fraud)

Provide structured output:
Risk Level: [0|1|2]
Key Pattern: [one-sentence description of the recurring pattern]
Confidence: [0.0-1.0]
"""


def build_risk_prompt(current_summary: str, retrieved_summaries: list[str]) -> str:
    """Build LLM input prompt for cross-session risk reasoning."""
    retrieved_text = "\n".join(
        [f"[Session {i+1}]: {s}" for i, s in enumerate(retrieved_summaries)]
    )
    return RISK_ASSESSMENT_PROMPT.format(
        current_session=current_summary,
        retrieved_sessions=retrieved_text,
    )


# ---------------------------------------------------------------------------
# Surrogate teacher (MLP approximation for reproducibility)
# ---------------------------------------------------------------------------
class LLMTeacherSurrogate(nn.Module):
    """
    Lightweight surrogate for the LLM teacher.
    Takes session embeddings (current + retrieved) and produces soft risk logits.
    Replace with actual LLM API calls in production.
    """

    def __init__(self, hidden_dim: int, k: int = 5, num_risk_levels: int = 3):
        super().__init__()
        # Attend over retrieved sessions to simulate LLM cross-session reasoning
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_risk_levels),
        )
        self.hidden_dim = hidden_dim

    @torch.no_grad()
    def forward(self, curr_emb: torch.Tensor, retrieved_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            curr_emb:      (B, H) — current session embedding
            retrieved_emb: (B, K, H) — retrieved historical embeddings
        Returns:
            teacher_logits: (B, num_risk_levels)
        """
        q = curr_emb.unsqueeze(1)  # (B, 1, H)
        attn_out, _ = self.cross_attn(q, retrieved_emb, retrieved_emb)
        attn_out = attn_out.squeeze(1)  # (B, H)
        combined = torch.cat([curr_emb, attn_out], dim=-1)
        return self.fusion(combined)


# ---------------------------------------------------------------------------
# KD loss
# ---------------------------------------------------------------------------
class CSVARDistillationLoss(nn.Module):
    """
    CS-VAR knowledge distillation loss.

    L_total = lambda_ce * L_CE(student, hard_label)
            + lambda_kd * T^2 * L_KD(student, teacher)

    Paper: LLM teacher output guides student via soft target distribution.
    """

    def __init__(self, temperature: float = 3.0, lambda_ce: float = 0.3, lambda_kd: float = 0.7):
        super().__init__()
        self.T = temperature
        self.lambda_ce = lambda_ce
        self.lambda_kd = lambda_kd

    def forward(self, student_logits, teacher_logits, hard_labels):
        ce = F.cross_entropy(student_logits, hard_labels)
        student_soft = F.log_softmax(student_logits / self.T, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.T, dim=-1)
        kd = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (self.T ** 2)
        return self.lambda_ce * ce + self.lambda_kd * kd
