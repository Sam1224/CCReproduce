"""
UNIVID model reproduction — PyTorch implementation.

Paper: UNIVID: Unified Vision-Language Model for Video Moderation (arXiv:2606.05748)
Authors: Kejuan Yang et al., ByteDance

Architecture summary (from paper):
  1. UNIVID Backbone — a VLM that generates policy-aware captions from video frames.
     We reproduce the captioning head and moderation head on top of a pretrained
     language model (using a small distilgpt2 as toy backbone).
  2. UNIVID-Lite — lightweight downstream model that consumes UNIVID captions to
     predict binary violation decisions (fast path).
  3. UNIVID-RAG — retrieval-augmented downstream model that recalls prior violative
     events and uses them to reduce leakage (slow path, handles edge cases).
  4. Trend Head — a lightweight head on cached UNIVID embeddings that detects
     emerging risk patterns via cluster assignments.

Formula references:
  Policy-aware caption loss:  L_cap = -Σ log P(c_t | c_{<t}, V, π)   [Eq. 1]
    where V = visual features, π = policy context
  Violation decision loss:    L_vio = BCE(σ(f(caption)), y)           [Eq. 2]
  Total loss:  L = λ_cap * L_cap + λ_vio * L_vio                      [Eq. 3]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from typing import Optional, Tuple, Dict


# ── UNIVID Backbone ───────────────────────────────────────────────────────────

class VisualProjection(nn.Module):
    """Projects visual features into the LM embedding space."""

    def __init__(self, visual_dim: int = 768, lm_dim: int = 768):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(visual_dim, lm_dim),
            nn.LayerNorm(lm_dim),
            nn.GELU(),
            nn.Linear(lm_dim, lm_dim),
        )

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        return self.proj(visual_features)


class PolicyConditioner(nn.Module):
    """
    Embeds policy context (which category to check) as a conditioning vector.
    In the paper, policy π is described textually and prepended as a prefix;
    here we implement it as a learned embedding over policy IDs.
    """

    def __init__(self, n_policies: int, lm_dim: int = 768):
        super().__init__()
        self.policy_embed = nn.Embedding(n_policies, lm_dim)
        self.gate = nn.Linear(lm_dim * 2, lm_dim)

    def forward(self, policy_ids: torch.Tensor, base_embed: torch.Tensor) -> torch.Tensor:
        policy_vec = self.policy_embed(policy_ids)           # (B, lm_dim)
        combined = torch.cat([base_embed, policy_vec], dim=-1)
        return torch.sigmoid(self.gate(combined)) * base_embed + policy_vec


class UNIVIDBackbone(nn.Module):
    """
    Toy UNIVID backbone.
    Real paper: large VLM (e.g., InternVL or similar) fine-tuned with
    policy-aware caption generation + moderation loss jointly.

    Here: distilgpt2-sized LM with visual adapter and policy conditioning.
    The caption is generated autoregressively; the violation score comes from
    a classification head on the final hidden state.

    Caption generation loss (Eq. 1):
        L_cap = - (1/T) Σ_t log P(c_t | c_{<t}, proj(V), embed(π))
    """

    def __init__(
        self,
        lm_model_name: str = "distilgpt2",
        visual_dim: int = 768,
        n_policies: int = 7,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lm = AutoModelForCausalLM.from_pretrained(lm_model_name)
        lm_dim = self.lm.config.hidden_size

        self.visual_proj = VisualProjection(visual_dim, lm_dim)
        self.policy_cond = PolicyConditioner(n_policies, lm_dim)
        self.dropout = nn.Dropout(dropout)

        # Violation head: binary decision from pooled hidden states
        # L_vio = BCE(σ(violation_head(h_pool)), y_violation)  [Eq. 2]
        self.violation_head = nn.Sequential(
            nn.Linear(lm_dim, lm_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lm_dim // 2, 1),
        )

        # Policy classification head (auxiliary)
        self.policy_head = nn.Linear(lm_dim, n_policies)

    def encode_caption(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run LM forward pass and return pooled representation."""
        outputs = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # Pool over non-padding positions
        hidden = outputs.hidden_states[-1]  # (B, T, D)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return pooled  # (B, D)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        visual_features: torch.Tensor,
        policy_label: Optional[torch.Tensor] = None,
        violation_label: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        Returns:
          - loss (if labels/violation_label provided)
          - violation_logit: (B,) — raw violation score
          - policy_logit:    (B, n_policies) — which policy category
          - pooled:          (B, D) — embedding for RAG / Trend Governance
        """
        # Visual + policy conditioning
        v_proj = self.visual_proj(visual_features)         # (B, D)

        if policy_label is not None:
            v_conditioned = self.policy_cond(policy_label, v_proj)
        else:
            v_conditioned = v_proj

        # Caption generation loss (causal LM loss over input tokens)
        # In real UNIVID the policy caption is decoded from visual+policy tokens;
        # here we use existing caption tokens as targets.
        lm_out = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids if labels is None else labels,
            output_hidden_states=True,
        )
        caption_loss = lm_out.loss  # L_cap

        # Pooled hidden state
        hidden = lm_out.hidden_states[-1]           # (B, T, D)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)  # (B, D)
        pooled = self.dropout(pooled + v_conditioned)

        # Violation head
        violation_logit = self.violation_head(pooled).squeeze(-1)  # (B,)
        policy_logit = self.policy_head(pooled)                     # (B, n_policies)

        loss = caption_loss
        if violation_label is not None:
            # L_vio = BCE(σ(violation_logit), y)   [Eq. 2]
            vio_loss = F.binary_cross_entropy_with_logits(
                violation_logit, violation_label
            )
            # L = λ_cap * L_cap + λ_vio * L_vio   [Eq. 3]
            loss = 0.5 * caption_loss + 0.5 * vio_loss

        if policy_label is not None:
            policy_loss = F.cross_entropy(policy_logit, policy_label)
            loss = loss + 0.1 * policy_loss

        return {
            "loss": loss,
            "violation_logit": violation_logit,
            "policy_logit": policy_logit,
            "pooled": pooled,
        }


# ── UNIVID-Lite ───────────────────────────────────────────────────────────────

class UNIVIDLite(nn.Module):
    """
    Lightweight downstream moderation head (Stage B, fast path).
    Consumes frozen UNIVID caption embeddings (pooled) and predicts
    binary violation decision.

    In the paper, UNIVID-Lite is a fine-tuned smaller model that
    ingests the UNIVID caption text and outputs a policy decision.
    Here we replicate the classification head over frozen embeddings.
    """

    def __init__(self, embed_dim: int = 768, n_policies: int = 7):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.policy_classifier = nn.Linear(embed_dim, n_policies)

    def forward(
        self,
        pooled: torch.Tensor,
        violation_label: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        logit = self.classifier(pooled).squeeze(-1)
        policy_logit = self.policy_classifier(pooled)

        loss = None
        if violation_label is not None:
            loss = F.binary_cross_entropy_with_logits(logit, violation_label)

        return {"loss": loss, "violation_logit": logit, "policy_logit": policy_logit}


# ── UNIVID-RAG ────────────────────────────────────────────────────────────────

class UNIVIDRag(nn.Module):
    """
    RAG-enhanced downstream model (Stage B, slow path for recall leakage).

    Pipeline:
      1. Encode query embedding (UNIVID pooled output for incoming video)
      2. Retrieve top-k similar past violation embeddings from FAISS index
      3. Fuse query + retrieved context to predict violation

    In the paper, historical violative events are indexed once and retrieved
    via approximate nearest-neighbor search. The retrieved captions provide
    context about known violation patterns, reducing leakage.

    L_rag = BCE(σ(f(concat([q, r_1, ..., r_k]))), y)
    """

    def __init__(self, embed_dim: int = 768, top_k: int = 3):
        super().__init__()
        self.top_k = top_k
        self.embed_dim = embed_dim

        # Context fusion: concat query + mean of top-k retrieved
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, 1),
        )

        # Retrieval index: toy in-memory version (real paper uses FAISS)
        self.register_buffer("index_embeddings", torch.zeros(0, embed_dim))
        self.register_buffer("index_labels", torch.zeros(0, dtype=torch.long))

    @torch.no_grad()
    def build_index(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """Build in-memory retrieval index from violation examples."""
        self.index_embeddings = embeddings.detach()
        self.index_labels = labels.detach()

    @torch.no_grad()
    def retrieve(self, query: torch.Tensor) -> torch.Tensor:
        """
        Retrieve top-k similar violation embeddings.
        Returns: (B, top_k, D) retrieved context embeddings.
        """
        if self.index_embeddings.shape[0] == 0:
            return torch.zeros(query.shape[0], self.top_k, self.embed_dim,
                               device=query.device)

        # Cosine similarity retrieval
        q_norm = F.normalize(query, dim=-1)                    # (B, D)
        idx_norm = F.normalize(self.index_embeddings, dim=-1)  # (N, D)
        sims = q_norm @ idx_norm.T                             # (B, N)
        topk_idx = sims.topk(
            min(self.top_k, sims.shape[1]), dim=-1
        ).indices                                              # (B, k)
        return self.index_embeddings[topk_idx]                 # (B, k, D)

    def forward(
        self,
        query_embed: torch.Tensor,
        violation_label: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        retrieved = self.retrieve(query_embed)              # (B, k, D)
        context = retrieved.mean(1)                         # (B, D)
        fused = torch.cat([query_embed, context], dim=-1)  # (B, 2D)
        logit = self.fusion(fused).squeeze(-1)             # (B,)

        loss = None
        if violation_label is not None:
            loss = F.binary_cross_entropy_with_logits(logit, violation_label)

        return {"loss": loss, "violation_logit": logit}


# ── Trend Governance Head ─────────────────────────────────────────────────────

class TrendGovernanceHead(nn.Module):
    """
    Stage C: Trend Governance using cached UNIVID embeddings.

    The paper caches UNIVID embeddings for all moderated videos and trains
    a small "trend head" that detects clusters of newly emerging violations
    by monitoring embedding distribution shifts over time windows.

    Here we implement a simplified version: a prototype-based anomaly
    detector that compares new embedding distributions to historical
    violation cluster centroids.

    Real paper formula: trend_score = max_k cos(e, μ_k)
    where μ_k are cluster centroids of historical violation embeddings.
    """

    def __init__(self, embed_dim: int = 768, n_clusters: int = 32):
        super().__init__()
        self.n_clusters = n_clusters
        self.embed_dim = embed_dim

        # Trainable cluster prototypes
        self.prototypes = nn.Parameter(
            torch.randn(n_clusters, embed_dim) * 0.01
        )
        # Threshold for trend detection
        self.trend_threshold = nn.Parameter(torch.tensor(0.7))

    def forward(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            embeddings: (B, D) — UNIVID pooled embeddings for incoming videos
        Returns:
            trend_score: (B,) — max similarity to any violation cluster
            is_trending: (B,) — bool, whether above threshold
        """
        e_norm = F.normalize(embeddings, dim=-1)         # (B, D)
        p_norm = F.normalize(self.prototypes, dim=-1)    # (K, D)

        # cos(e, μ_k) for each prototype
        sims = e_norm @ p_norm.T                         # (B, K)
        trend_score = sims.max(dim=-1).values            # (B,)
        is_trending = trend_score > torch.sigmoid(self.trend_threshold)

        return {
            "trend_score": trend_score,
            "is_trending": is_trending,
            "similarities": sims,
        }

    def update_prototypes(self, violation_embeds: torch.Tensor, momentum: float = 0.9):
        """
        EMA update of cluster prototypes using new violation embeddings.
        In the real paper this is done via online k-means / EM over cached embeddings.
        """
        with torch.no_grad():
            e_norm = F.normalize(violation_embeds, dim=-1)      # (N, D)
            p_norm = F.normalize(self.prototypes.data, dim=-1)  # (K, D)
            sims = e_norm @ p_norm.T                             # (N, K)
            assignments = sims.argmax(dim=-1)                   # (N,)
            for k in range(self.n_clusters):
                mask = assignments == k
                if mask.sum() > 0:
                    cluster_mean = violation_embeds[mask].mean(0)
                    self.prototypes.data[k] = (
                        momentum * self.prototypes.data[k]
                        + (1 - momentum) * cluster_mean
                    )


# ── Full UNIVID System ────────────────────────────────────────────────────────

class UNIVIDSystem(nn.Module):
    """
    Full UNIVID system combining backbone + Lite + RAG + Trend heads.

    Inference pipeline (paper Figure 2):
      Stage A (Risk Filter):     backbone generates caption + rough violation score
      Stage B (Moderation):      Lite (fast path) + RAG (recall leakage)
      Stage C (Trend):           TrendHead monitors embedding clusters
    """

    def __init__(
        self,
        lm_model_name: str = "distilgpt2",
        visual_dim: int = 768,
        n_policies: int = 7,
    ):
        super().__init__()
        self.backbone = UNIVIDBackbone(lm_model_name, visual_dim, n_policies)
        lm_dim = self.backbone.lm.config.hidden_size
        self.lite = UNIVIDLite(lm_dim, n_policies)
        self.rag = UNIVIDRag(lm_dim)
        self.trend = TrendGovernanceHead(lm_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        visual_features: torch.Tensor,
        policy_label: Optional[torch.Tensor] = None,
        violation_label: Optional[torch.Tensor] = None,
        stage: str = "backbone",
    ) -> Dict[str, torch.Tensor]:
        backbone_out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_features=visual_features,
            policy_label=policy_label,
            violation_label=violation_label,
        )
        pooled = backbone_out["pooled"].detach()

        if stage == "lite":
            lite_out = self.lite(pooled, violation_label)
            return {**backbone_out, **{f"lite_{k}": v for k, v in lite_out.items()}}
        elif stage == "rag":
            rag_out = self.rag(pooled, violation_label)
            return {**backbone_out, **{f"rag_{k}": v for k, v in rag_out.items()}}
        elif stage == "trend":
            trend_out = self.trend(pooled)
            return {**backbone_out, **{f"trend_{k}": v for k, v in trend_out.items()}}
        else:
            return backbone_out

    @torch.no_grad()
    def moderate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        visual_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Full three-stage inference pipeline.
        Returns combined violation decision and trend signal.
        """
        backbone_out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_features=visual_features,
        )
        pooled = backbone_out["pooled"]

        # Stage A: backbone risk score
        risk_score = torch.sigmoid(backbone_out["violation_logit"])

        # Stage B: Lite fast path
        lite_out = self.lite(pooled)
        lite_score = torch.sigmoid(lite_out["violation_logit"])

        # Stage B: RAG recall
        rag_out = self.rag(pooled)
        rag_score = torch.sigmoid(rag_out["violation_logit"])

        # Stage C: Trend signal
        trend_out = self.trend(pooled)

        # Ensemble: max of risk, lite, rag (conservative — reduce leakage)
        ensemble_score = torch.stack([risk_score, lite_score, rag_score]).max(0).values
        decision = ensemble_score > 0.5

        return {
            "risk_score": risk_score,
            "lite_score": lite_score,
            "rag_score": rag_score,
            "ensemble_score": ensemble_score,
            "decision": decision,
            "trend_score": trend_out["trend_score"],
            "is_trending": trend_out["is_trending"],
            "policy_pred": backbone_out["policy_logit"].argmax(-1),
        }
