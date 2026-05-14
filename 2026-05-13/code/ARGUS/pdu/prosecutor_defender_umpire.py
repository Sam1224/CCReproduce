"""
Prosecutor-Defender-Umpire (PDU) Architecture — Stage II of ARGUS.

From paper §3.2 (Adversarial Label Rectification):

    "We propose a 'Prosecutor-Defender-Umpire' (PDU) architecture to resolve
     conflicts between stale historical labels and new regulatory mandates.

     Prosecutor: argues that the sample violates the NEW policy.
                 Evidence: new policy keywords, violation patterns.

     Defender:   argues that the sample complies (consistent with HISTORICAL label).
                 Evidence: surface-level compliance signals, historical precedent.

     Umpire:     reviews the arguments of both sides, weighs evidence, and produces
                 a final rectified label + confidence.

     The adversarial dialectic forces explicit reasoning about ambiguous 'gray-area'
     samples, improving label quality beyond what either agent alone can achieve."

PDU Formulation (paper, inferred from §3.2):
    For sample x with historical label y_h and candidate new label y_n:
        prosecutor_score(x) = f_P(x, y_n, policy_new)      ∈ [0, 1]
        defender_score(x)   = f_D(x, y_h, policy_history)  ∈ [0, 1]
        umpire_label(x)     = argmax_{y} g_U(prosecutor_score, defender_score, x)

    Where g_U learns to weigh the two adversarial signals.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


LABEL_SPACE = ["compliant", "violation"]


POLICY_VIOLATION_KEYWORDS = {
    "education_anxiety": [
        "left behind", "fall behind", "fail", "lag", "danger",
        "risk", "lose", "struggle", "must", "urgent", "elite", "fear"
    ],
    "health_anxiety": [
        "fading", "disease", "sick", "pain", "danger", "risk",
        "guaranteed", "miracle", "doctor-approved", "sharp", "old"
    ],
    "appearance_anxiety": [
        "ugly", "fat", "thin", "aging", "wrinkle", "old", "bad skin",
        "look old", "embarrassing", "fix", "miracle"
    ],
}

ALL_VIOLATION_KEYWORDS = set()
for kws in POLICY_VIOLATION_KEYWORDS.values():
    ALL_VIOLATION_KEYWORDS.update(kws)

COMPLIANCE_SIGNALS = [
    "sale", "discount", "delivery", "quality", "premium", "fresh",
    "offer", "deal", "new", "available", "best", "professional"
]


@dataclass
class PDUResult:
    sample_id: str
    ad_text: str
    historical_label: str
    rectified_label: str
    prosecutor_score: float
    defender_score: float
    umpire_confidence: float
    prosecutor_evidence: str
    defender_evidence: str
    umpire_reasoning: str
    is_gray_area: bool


class Prosecutor(nn.Module):
    """
    Argues that the ad violates the NEW policy.
    Score ∈ [0, 1]: 1 = strong violation signal.
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.fc = nn.Linear(1, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def _keyword_score(self, text: str) -> float:
        text_lower = text.lower()
        matches = sum(1 for kw in ALL_VIOLATION_KEYWORDS if kw in text_lower)
        return min(matches / max(len(ALL_VIOLATION_KEYWORDS) * 0.1, 1), 1.0)

    def _find_evidence(self, text: str) -> str:
        text_lower = text.lower()
        found = [kw for kw in ALL_VIOLATION_KEYWORDS if kw in text_lower]
        return f"Violation keywords found: {found[:3]}" if found else "No explicit violation keywords"

    def forward(self, texts: list) -> tuple:
        scores = [self._keyword_score(t) for t in texts]
        evidences = [self._find_evidence(t) for t in texts]
        score_t = torch.tensor(scores).unsqueeze(-1)
        # Toy neural layer over the heuristic score
        refined = torch.sigmoid(self.out(F.gelu(self.fc(score_t)))).squeeze(-1)
        return refined, evidences


class Defender(nn.Module):
    """
    Argues that the ad complies with historical policy.
    Score ∈ [0, 1]: 1 = strong compliance signal.
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.fc = nn.Linear(1, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def _compliance_score(self, text: str, historical_label: str) -> float:
        text_lower = text.lower()
        match = sum(1 for s in COMPLIANCE_SIGNALS if s in text_lower)
        base = match / max(len(COMPLIANCE_SIGNALS) * 0.2, 1)
        if historical_label == "compliant":
            base = min(base + 0.3, 1.0)
        return min(base, 1.0)

    def _find_evidence(self, text: str, historical_label: str) -> str:
        text_lower = text.lower()
        found = [s for s in COMPLIANCE_SIGNALS if s in text_lower]
        return (
            f"Historical label='{historical_label}'. Compliance signals: {found[:3]}"
            if found else f"Historical label='{historical_label}'. No compliance signals found."
        )

    def forward(self, texts: list, historical_labels: list) -> tuple:
        scores = [self._compliance_score(t, h) for t, h in zip(texts, historical_labels)]
        evidences = [self._find_evidence(t, h) for t, h in zip(texts, historical_labels)]
        score_t = torch.tensor(scores).unsqueeze(-1)
        refined = torch.sigmoid(self.out(F.gelu(self.fc(score_t)))).squeeze(-1)
        return refined, evidences


class Umpire(nn.Module):
    """
    Arbitrates between Prosecutor and Defender scores.

    Umpire decision (paper §3.2, inferred):
        input: [prosecutor_score, defender_score, |p-d|, p*d]
        output: P(violation) ∈ [0, 1]
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.gray_area_threshold = 0.15  # |p - d| < threshold → gray area

    def forward(
        self,
        prosecutor_scores: torch.Tensor,
        defender_scores: torch.Tensor,
    ) -> tuple:
        diff = (prosecutor_scores - defender_scores).abs()
        product = prosecutor_scores * defender_scores
        x = torch.stack([prosecutor_scores, defender_scores, diff, product], dim=-1)
        violation_prob = self.net(x).squeeze(-1)

        labels = ["violation" if p > 0.5 else "compliant" for p in violation_prob.tolist()]
        confidence = [abs(p.item() - 0.5) * 2 for p in violation_prob]
        is_gray = [d.item() < self.gray_area_threshold for d in diff]
        return labels, confidence, is_gray, violation_prob


class PDUSystem(nn.Module):
    """Full Prosecutor-Defender-Umpire system."""

    def __init__(self):
        super().__init__()
        self.prosecutor = Prosecutor()
        self.defender = Defender()
        self.umpire = Umpire()

    def rectify(
        self,
        sample_ids: list,
        texts: list,
        historical_labels: list,
    ) -> list:
        p_scores, p_evidences = self.prosecutor(texts)
        d_scores, d_evidences = self.defender(texts, historical_labels)
        u_labels, u_confs, is_gray, _ = self.umpire(p_scores, d_scores)

        results = []
        for i in range(len(texts)):
            reasoning = (
                f"Prosecutor({p_scores[i].item():.3f}) vs "
                f"Defender({d_scores[i].item():.3f}) → "
                f"Umpire rectifies to '{u_labels[i]}' "
                f"(conf={u_confs[i]:.3f}, gray={is_gray[i]})"
            )
            results.append(PDUResult(
                sample_id=sample_ids[i],
                ad_text=texts[i],
                historical_label=historical_labels[i],
                rectified_label=u_labels[i],
                prosecutor_score=p_scores[i].item(),
                defender_score=d_scores[i].item(),
                umpire_confidence=u_confs[i],
                prosecutor_evidence=p_evidences[i],
                defender_evidence=d_evidences[i],
                umpire_reasoning=reasoning,
                is_gray_area=is_gray[i],
            ))
        return results

    def train_on_rectified(
        self,
        texts: list,
        rectified_labels: list,
        n_steps: int = 5,
    ) -> float:
        """Fine-tune PDU components on rectified labels (Stage II training)."""
        label2idx = {"compliant": 0, "violation": 1}
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        total_loss = 0.0
        for step in range(n_steps):
            p_scores, _ = self.prosecutor(texts)
            d_scores, _ = self.defender(texts, ["compliant"] * len(texts))
            _, _, _, violation_probs = self.umpire(p_scores, d_scores)
            labels = torch.tensor(
                [label2idx.get(l, 0) for l in rectified_labels], dtype=torch.float
            )
            loss = F.binary_cross_entropy(violation_probs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / n_steps
