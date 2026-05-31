"""
Reproduction: Thinking Broad, Acting Fast
"Latent Reasoning Distillation from Multi-Perspective Chain-of-Thought for E-Commerce Relevance"
arXiv: 2601.21611  |  WWW 2026

Architecture:
  Teacher LLM → Multi-Perspective CoT Generation
  Student Model ← Latent Reasoning Distillation

This module implements:
  1. Multi-Perspective CoT prompt templates
  2. Teacher LLM multi-perspective reasoning extractor
  3. Student model with latent reasoning alignment loss
  4. Full distillation training pipeline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from typing import List, Dict, Optional, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# 1. Multi-Perspective CoT Prompt Templates
# ---------------------------------------------------------------------------

ECOM_RELEVANCE_PERSPECTIVES = [
    {
        "name": "semantic_match",
        "description": "Evaluate whether the product's title and description semantically match the user query.",
        "prompt_template": (
            "Query: {query}\nProduct: {product_title}\n"
            "From a SEMANTIC perspective, does this product match the query? "
            "Consider synonyms, related terms, and category alignment. Reason step-by-step:\n"
        ),
    },
    {
        "name": "user_intent",
        "description": "Infer the user's underlying purchase intent and check if the product satisfies it.",
        "prompt_template": (
            "Query: {query}\nProduct: {product_title}\n"
            "What is the user's PURCHASE INTENT behind this query? "
            "Does this product satisfy that intent? Reason step-by-step:\n"
        ),
    },
    {
        "name": "attribute_match",
        "description": "Check alignment of specific product attributes (size, color, material, brand) with query constraints.",
        "prompt_template": (
            "Query: {query}\nProduct: {product_title}\nProduct Attributes: {product_attrs}\n"
            "From an ATTRIBUTE perspective, do the specific product attributes "
            "(size, color, material, brand, etc.) match any constraints in the query? Reason step-by-step:\n"
        ),
    },
    {
        "name": "long_tail_coverage",
        "description": "Assess if rare/specialized query terms are covered by the product.",
        "prompt_template": (
            "Query: {query}\nProduct: {product_title}\n"
            "This might be a long-tail or specialized query. Does the product specifically address "
            "the niche requirement? Consider domain-specific terminology. Reason step-by-step:\n"
        ),
    },
]


def build_multi_perspective_prompts(
    query: str,
    product_title: str,
    product_attrs: str = "",
    perspectives: List[Dict] = ECOM_RELEVANCE_PERSPECTIVES,
) -> List[str]:
    """Build prompts for each CoT perspective."""
    prompts = []
    for p in perspectives:
        prompt = p["prompt_template"].format(
            query=query,
            product_title=product_title,
            product_attrs=product_attrs,
        )
        prompts.append(prompt)
    return prompts


# ---------------------------------------------------------------------------
# 2. Teacher LLM Multi-Perspective Reasoning Generator
# ---------------------------------------------------------------------------

class TeacherLLM:
    """
    Wraps a pre-trained LLM (e.g., LLaMA-3, Qwen2, GPT-4) to generate
    multi-perspective Chain-of-Thought reasoning for query-product pairs.
    """

    def __init__(self, model_name: str = "meta-llama/Llama-3-8B-Instruct", device: str = "cuda"):
        self.device = device
        # In production: load actual LLM
        # For reproduction: we use a placeholder that simulates CoT generation
        self.model_name = model_name
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        print(f"[TeacherLLM] Initialized with model: {model_name}")

    def generate_reasoning(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.3,
    ) -> str:
        """
        Generate CoT reasoning for a given prompt.

        In production: call actual LLM inference.
        Here: return a structured placeholder demonstrating the expected format.
        """
        # Pseudocode for actual implementation:
        # inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        # with torch.no_grad():
        #     outputs = self.model.generate(
        #         **inputs,
        #         max_new_tokens=max_new_tokens,
        #         temperature=temperature,
        #         do_sample=True,
        #     )
        # return self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        return f"[Simulated CoT reasoning for prompt: {prompt[:50]}...] The product semantically aligns because..."

    def generate_multi_perspective_reasoning(
        self,
        query: str,
        product_title: str,
        product_attrs: str = "",
    ) -> Dict[str, str]:
        """
        Generate reasoning chains from all perspectives for a query-product pair.
        Returns: {perspective_name: reasoning_text}
        """
        prompts = build_multi_perspective_prompts(query, product_title, product_attrs)
        reasoning_by_perspective = {}
        for perspective, prompt in zip(ECOM_RELEVANCE_PERSPECTIVES, prompts):
            reasoning = self.generate_reasoning(prompt)
            reasoning_by_perspective[perspective["name"]] = reasoning
        return reasoning_by_perspective

    def score_relevance(self, reasoning_traces: Dict[str, str]) -> float:
        """
        Aggregate multi-perspective reasoning into a final relevance score.
        In practice: prompt the teacher to output a score after reasoning.
        """
        # Pseudocode:
        # aggregate_prompt = build_aggregation_prompt(reasoning_traces)
        # score_text = self.generate_reasoning(aggregate_prompt)
        # return parse_score(score_text)
        return 0.75  # placeholder


# ---------------------------------------------------------------------------
# 3. Student Model with Latent Reasoning Alignment
# ---------------------------------------------------------------------------

class LatentReasoningEncoder(nn.Module):
    """
    Encodes latent reasoning representations from the teacher's multi-perspective CoT.
    Formula (Section 3.2 of paper):
        z_teacher = PoolingEncode(CoT_1, CoT_2, ..., CoT_K)
        where K = number of perspectives
    """

    def __init__(self, hidden_dim: int = 768, num_perspectives: int = 4):
        super().__init__()
        self.perspective_projectors = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_perspectives)
        ])
        self.aggregation = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=8, batch_first=True
        )
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, perspective_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            perspective_embeddings: (batch, num_perspectives, hidden_dim)
        Returns:
            aggregated: (batch, hidden_dim) — unified latent reasoning representation
        """
        projected = []
        for i, proj in enumerate(self.perspective_projectors):
            projected.append(proj(perspective_embeddings[:, i, :]))
        # Stack: (batch, num_perspectives, hidden_dim)
        stacked = torch.stack(projected, dim=1)
        # Cross-perspective attention aggregation
        attended, _ = self.aggregation(stacked, stacked, stacked)
        # Mean pool over perspectives
        aggregated = attended.mean(dim=1)
        return self.output_proj(aggregated)


class StudentRelevanceModel(nn.Module):
    """
    Fast student model for e-commerce query-product relevance scoring.

    Architecture:
        Query Encoder → q_repr
        Product Encoder → p_repr
        Interaction Layer → interaction features
        Latent Reasoning Head → z_student (aligned to teacher's z_teacher)
        Relevance Scorer → score

    Distillation Loss (Eq. 5 in paper):
        L_total = L_relevance + λ * L_latent_alignment
        L_latent_alignment = MSE(z_student, z_teacher_detached)
    """

    def __init__(
        self,
        encoder_name: str = "bert-base-uncased",
        hidden_dim: int = 768,
        num_perspectives: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Shared encoder for query and product
        # self.encoder = AutoModel.from_pretrained(encoder_name)

        # Placeholder: mock encoder that produces (batch, seq_len, hidden_dim)
        self.hidden_dim = hidden_dim
        self.mock_encoder_proj = nn.Linear(hidden_dim, hidden_dim)

        # Interaction layer: computes fine-grained query-product interactions
        self.interaction = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # [q, p, q-p]
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Latent reasoning head: student learns to produce teacher-aligned representations
        self.latent_reasoning_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Final relevance scorer
        self.relevance_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.num_perspectives = num_perspectives

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Encode text to CLS representation.
        Returns: (batch, hidden_dim)
        """
        # Pseudocode:
        # outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # return outputs.last_hidden_state[:, 0, :]  # CLS token
        batch_size = input_ids.shape[0]
        return torch.randn(batch_size, self.hidden_dim, device=input_ids.device)

    def forward(
        self,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        product_input_ids: torch.Tensor,
        product_attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            relevance_score: (batch,)
            latent_z: (batch, hidden_dim) — for alignment with teacher
        """
        q_repr = self.encode(query_input_ids, query_attention_mask)
        p_repr = self.encode(product_input_ids, product_attention_mask)

        # Element-wise difference for fine-grained matching
        diff = q_repr - p_repr
        combined = torch.cat([q_repr, p_repr, diff], dim=-1)
        interaction_feat = self.interaction(combined)

        # Latent reasoning representation (aligned to teacher via distillation)
        latent_z = self.latent_reasoning_head(interaction_feat)

        # Final scoring
        score_input = torch.cat([interaction_feat, latent_z], dim=-1)
        relevance_score = self.relevance_scorer(score_input).squeeze(-1)

        return relevance_score, latent_z


# ---------------------------------------------------------------------------
# 4. Latent Reasoning Distillation Loss
# ---------------------------------------------------------------------------

class LatentReasoningDistillationLoss(nn.Module):
    """
    Combined distillation loss:
        L = L_relevance + lambda * L_latent_alignment

    L_relevance: cross-entropy on binary relevance labels
    L_latent_alignment: MSE between student z and teacher z (detached)

    Eq. (5) from paper.
    """

    def __init__(self, lambda_distill: float = 0.5):
        super().__init__()
        self.lambda_distill = lambda_distill
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        student_scores: torch.Tensor,
        student_latent: torch.Tensor,
        teacher_latent: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            student_scores: (batch,) — raw relevance logits from student
            student_latent: (batch, hidden_dim) — student latent reasoning repr
            teacher_latent: (batch, hidden_dim) — teacher latent repr (detached)
            labels: (batch,) — binary relevance labels (0 or 1)
        Returns:
            losses dict with total, relevance, alignment components
        """
        # Relevance classification loss
        l_relevance = self.bce_loss(student_scores, labels.float())

        # Latent alignment loss: student learns to mimic teacher's reasoning space
        # teacher_latent is detached — we do not backprop through the teacher
        l_align = self.mse_loss(student_latent, teacher_latent.detach())

        # Total loss
        l_total = l_relevance + self.lambda_distill * l_align

        return {
            "total": l_total,
            "relevance": l_relevance,
            "alignment": l_align,
        }


# ---------------------------------------------------------------------------
# 5. Teacher Latent Representation Extractor
# ---------------------------------------------------------------------------

class TeacherLatentExtractor(nn.Module):
    """
    Encodes teacher CoT reasoning text into latent representations.

    For each perspective's reasoning text:
        embed = TextEncoder(reasoning_text)
    Then aggregate across perspectives via LatentReasoningEncoder.
    """

    def __init__(self, text_encoder_name: str = "bert-base-uncased", hidden_dim: int = 768):
        super().__init__()
        # self.text_encoder = AutoModel.from_pretrained(text_encoder_name)
        self.hidden_dim = hidden_dim
        self.reasoning_aggregator = LatentReasoningEncoder(
            hidden_dim=hidden_dim, num_perspectives=len(ECOM_RELEVANCE_PERSPECTIVES)
        )

    def encode_reasoning_text(self, text: str) -> torch.Tensor:
        """
        Encode a single reasoning text string to a fixed-size vector.
        Returns: (1, hidden_dim)
        """
        # Pseudocode:
        # inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        # with torch.no_grad():
        #     outputs = self.text_encoder(**inputs)
        # return outputs.last_hidden_state[:, 0, :]
        return torch.randn(1, self.hidden_dim)

    def forward(self, reasoning_dict: Dict[str, str]) -> torch.Tensor:
        """
        Args:
            reasoning_dict: {perspective_name: reasoning_text}
        Returns:
            teacher_latent: (1, hidden_dim)
        """
        perspective_embeddings = []
        for perspective in ECOM_RELEVANCE_PERSPECTIVES:
            reasoning_text = reasoning_dict.get(perspective["name"], "")
            embedding = self.encode_reasoning_text(reasoning_text)
            perspective_embeddings.append(embedding)

        # Stack: (1, num_perspectives, hidden_dim)
        stacked = torch.stack(perspective_embeddings, dim=1)
        return self.reasoning_aggregator(stacked)
