"""
ARGUS — Model components
Implements the three-stage framework from arXiv 2605.02200.

Stage I  — Policy Seeding:       VLM with new policy context for initial perception
Stage II — Adversarial Label Rectification: Prosecutor-Defender-Umpire (PDU) architecture
Stage III — Latent Knowledge Discovery:    Hard negative synthesis via tripartite discussion

The VLM backbone is abstracted; swap in any HuggingFace multimodal model.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    AutoTokenizer,
    AutoModel,
)


# ---------------------------------------------------------------------------
# Lightweight VLM Wrapper (backbone agnostic)
# ---------------------------------------------------------------------------

class PolicyVLM(nn.Module):
    """
    Vision-Language Model backbone for ad governance.

    Wraps a HuggingFace VLM (e.g. Qwen2-VL, LLaVA, InternVL) and exposes
    a classify() interface that returns a binary label + reasoning chain.

    Paper: VLM is trained/fine-tuned with policy context in Stage I,
    and further adapted via RL reward in Stage III.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
        num_labels: int = 2,
        use_lora: bool = True,
        device: str = "cpu",
    ):
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.device = device

        # In production: load actual VLM
        # self.processor = AutoProcessor.from_pretrained(model_name)
        # self.vlm = AutoModelForVision2Seq.from_pretrained(model_name)

        # Toy backbone: simple vision + text encoder
        self.vision_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(3 * 7 * 7, 512),
            nn.GELU(),
            nn.Linear(512, 256),
        )

        # Simple text encoder (character n-gram hashing as toy)
        self.text_proj = nn.Linear(256, 256)

        # Fusion + classification head
        self.fusion = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_labels),
        )

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Toy text encoder: hash characters to fixed-dim vector."""
        batch = []
        for text in texts:
            vec = torch.zeros(256)
            for i, ch in enumerate(text[:256]):
                vec[i % 256] += ord(ch) / 128.0
            batch.append(vec)
        return torch.stack(batch).to(self.device)

    def forward(
        self,
        images: torch.Tensor,
        texts: List[str],
    ) -> Dict[str, torch.Tensor]:
        vis_feats = self.vision_encoder(images)         # (B, 256)
        txt_feats = self.encode_text(texts)             # (B, 256)
        txt_feats = self.text_proj(txt_feats)
        fused = torch.cat([vis_feats, txt_feats], dim=-1)  # (B, 512)
        logits = self.fusion(fused)                    # (B, 2)
        return {"logits": logits, "vis_feats": vis_feats, "txt_feats": txt_feats}

    def classify(
        self,
        images: torch.Tensor,
        texts: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.forward(images, texts)
        probs = F.softmax(out["logits"], dim=-1)
        preds = probs.argmax(dim=-1)
        return preds, probs


# ---------------------------------------------------------------------------
# Stage II: Prosecutor-Defender-Umpire (PDU) Architecture
# ---------------------------------------------------------------------------

@dataclass
class PDUResult:
    """Result of one PDU round."""
    prosecutor_verdict: int      # 0 or 1
    defender_verdict: int        # 0 or 1
    umpire_verdict: int          # final rectified label
    prosecutor_reasoning: str
    defender_reasoning: str
    umpire_reasoning: str
    confidence: float            # umpire confidence score


class ProsecutorAgent:
    """
    Strict regulatory inspector — identifies potential violations.

    Paper: "The Prosecutor agent is prompted to act as a strict regulatory
    inspector, with the objective to identify any potential violation of the
    newly emerging policy, no matter how subtle."
    """

    def __init__(self, vlm: PolicyVLM, policy_knowledge_base=None):
        self.vlm = vlm
        self.kb = policy_knowledge_base

    def judge(
        self,
        image: torch.Tensor,
        text: str,
        policy_context: str = "",
    ) -> Tuple[int, str]:
        """
        Returns (verdict, reasoning).
        verdict: 1 = violation, 0 = compliant (from strict inspector perspective).
        """
        # In production: construct prompt with policy_context and call VLM
        # Toy: use VLM classification with slight violation bias
        with torch.no_grad():
            img_batch = image.unsqueeze(0)
            preds, probs = self.vlm.classify(img_batch, [text])
        pred = preds[0].item()
        prob_violation = probs[0, 1].item()

        # Prosecutor applies stricter threshold (lower threshold for violation)
        verdict = 1 if prob_violation > 0.35 else 0
        reasoning = (
            f"[Prosecutor] Violation probability={prob_violation:.3f}. "
            f"Policy clause potentially triggered: educational anxiety restriction. "
            f"Verdict: {'VIOLATION' if verdict else 'COMPLIANT'}."
        )
        return verdict, reasoning


class DefenderAgent:
    """
    Sophisticated legal counsel — provides benign interpretations.

    Paper: "The Defender acts as 'sophisticated legal counsel' ... providing
    alternative, benign interpretations for every point of contention raised
    by the Prosecutor, such as interpreting an exaggerated claim as 'artistic
    hyperbole' or a high-pressure countdown as 'legitimate seasonal promotion'."
    """

    def __init__(self, vlm: PolicyVLM):
        self.vlm = vlm

    def judge(
        self,
        image: torch.Tensor,
        text: str,
        prosecutor_reasoning: str,
    ) -> Tuple[int, str]:
        """Returns (verdict, reasoning). Defender argues for compliance."""
        with torch.no_grad():
            img_batch = image.unsqueeze(0)
            preds, probs = self.vlm.classify(img_batch, [text])
        prob_violation = probs[0, 1].item()

        # Defender applies lenient threshold (higher threshold for violation)
        verdict = 1 if prob_violation > 0.70 else 0
        reasoning = (
            f"[Defender] Challenging prosecutor finding. "
            f"The language used can be interpreted as legitimate encouragement "
            f"rather than anxiety-inducing content. "
            f"Violation probability under lenient reading={prob_violation:.3f}. "
            f"Verdict: {'VIOLATION' if verdict else 'COMPLIANT'}."
        )
        return verdict, reasoning


class UmpireAgent:
    """
    Neutral adjudicator — resolves prosecutor vs. defender conflicts.

    Paper: "A neutral Umpire VLM then adjudicates these conflicting
    Chains-of-Thought, incorporating RAG-enhanced policy knowledge to
    rectify labels and provide high-fidelity rewards for reinforcement learning."
    """

    def __init__(self, vlm: PolicyVLM, policy_knowledge_base=None):
        self.vlm = vlm
        self.kb = policy_knowledge_base

    def adjudicate(
        self,
        image: torch.Tensor,
        text: str,
        prosecutor_verdict: int,
        prosecutor_reasoning: str,
        defender_verdict: int,
        defender_reasoning: str,
    ) -> Tuple[int, str, float]:
        """Returns (final_verdict, reasoning, confidence)."""

        # Retrieve RAG policy context
        rag_context = ""
        if self.kb is not None:
            retrieved = self.kb.retrieve(text, top_k=3)
            rag_context = self.kb.format_for_prompt(retrieved)

        # In production: call VLM with full PDU context + RAG
        with torch.no_grad():
            img_batch = image.unsqueeze(0)
            preds, probs = self.vlm.classify(img_batch, [text])
        prob_violation = probs[0, 1].item()

        # Umpire takes weighted combination; resolves conflict
        if prosecutor_verdict == defender_verdict:
            final_verdict = prosecutor_verdict
            confidence = 0.9
        else:
            # Conflict: umpire uses neutral VLM score
            final_verdict = 1 if prob_violation > 0.50 else 0
            confidence = abs(prob_violation - 0.50) * 2

        reasoning = (
            f"[Umpire] Prosecutor says {prosecutor_verdict}, Defender says {defender_verdict}. "
            f"RAG policy context: {rag_context[:100]}... "
            f"Neutral VLM score={prob_violation:.3f}. "
            f"Final verdict: {'VIOLATION' if final_verdict else 'COMPLIANT'} "
            f"(confidence={confidence:.3f})."
        )
        return final_verdict, reasoning, confidence


class PDUArchitecture:
    """
    Orchestrates the full Prosecutor-Defender-Umpire pipeline.

    Paper Stage II: Adversarial Label Rectification.
    Returns rectified labels and RL reward signals.
    """

    def __init__(
        self,
        vlm: PolicyVLM,
        policy_knowledge_base=None,
    ):
        self.prosecutor = ProsecutorAgent(vlm, policy_knowledge_base)
        self.defender = DefenderAgent(vlm)
        self.umpire = UmpireAgent(vlm, policy_knowledge_base)

    def rectify(
        self,
        image: torch.Tensor,
        text: str,
        stale_label: Optional[int] = None,
    ) -> PDUResult:
        """Run full PDU cycle and return rectified result."""
        p_verdict, p_reason = self.prosecutor.judge(image, text)
        d_verdict, d_reason = self.defender.judge(image, text, p_reason)
        u_verdict, u_reason, confidence = self.umpire.adjudicate(
            image, text, p_verdict, p_reason, d_verdict, d_reason
        )
        return PDUResult(
            prosecutor_verdict=p_verdict,
            defender_verdict=d_verdict,
            umpire_verdict=u_verdict,
            prosecutor_reasoning=p_reason,
            defender_reasoning=d_reason,
            umpire_reasoning=u_reason,
            confidence=confidence,
        )

    def compute_rl_reward(
        self,
        pdu_result: PDUResult,
        model_pred: int,
    ) -> float:
        """
        Compute RL reward based on PDU umpire verdict.

        Paper: "provide high-fidelity rewards for reinforcement learning"
        Reward shaped by umpire confidence to handle gray areas.
        """
        correct = (model_pred == pdu_result.umpire_verdict)
        base_reward = 1.0 if correct else -1.0
        # Scale by umpire confidence — high-confidence corrections get stronger signal
        return base_reward * pdu_result.confidence


# ---------------------------------------------------------------------------
# Stage III: Latent Knowledge Discovery — Hard Sample Synthesis
# ---------------------------------------------------------------------------

class HardSampleMiner:
    """
    Synthesizes gray-area adversarial samples via tripartite dialectical discussion.

    Paper Stage III: "employs a tripartite dialectical discussion to unearth
    sophisticated, 'gray-area' violations."
    """

    def __init__(self, pdu: PDUArchitecture):
        self.pdu = pdu

    def is_hard_sample(self, pdu_result: PDUResult) -> bool:
        """A sample is 'hard' if PDU agents disagree."""
        return pdu_result.prosecutor_verdict != pdu_result.defender_verdict

    def generate_adversarial_variant(self, text: str) -> str:
        """
        Generate a subtle adversarial variant of a compliant ad text.

        Production: call a generation LLM with targeted adversarial prompting.
        Toy: simple template substitution to create borderline examples.
        """
        borderline_patterns = [
            ("help", "ensure your child doesn't fall behind"),
            ("good", "better than others"),
            ("effective", "proven to give results others won't"),
            ("quality", "quality that separates winners from the rest"),
        ]
        result = text.lower()
        for src, tgt in borderline_patterns:
            if src in result:
                result = result.replace(src, tgt, 1)
                break
        return result

    def mine_hard_samples(
        self,
        dataset,
        vlm: PolicyVLM,
        max_samples: int = 100,
    ) -> List[Dict]:
        """
        Identify and synthesize hard adversarial samples from dataset.
        Returns augmented samples for Stage III fine-tuning.
        """
        hard_samples = []
        for i, sample in enumerate(dataset):
            if i >= max_samples:
                break
            image = sample["image"]
            text = sample["text"]
            pdu_result = self.pdu.rectify(image, text)

            if self.is_hard_sample(pdu_result):
                variant_text = self.generate_adversarial_variant(text)
                hard_samples.append({
                    "original_text": text,
                    "adversarial_text": variant_text,
                    "umpire_label": pdu_result.umpire_verdict,
                    "confidence": pdu_result.confidence,
                    "reasoning": pdu_result.umpire_reasoning,
                })

        return hard_samples
