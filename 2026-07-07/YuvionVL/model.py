"""
YuvionVL — Model Architecture

Implements:
1. YuvionVLModel: core MLLM backbone (simplified) with safety classification head
2. C2FT loss: Confuse-then-Contrast Fine-Tuning
3. Safety gate for inference-time policy enforcement

Paper: "Yuvion VL: A Multimodal Foundation Model for Adversarial Content and AI Safety"
arXiv: https://arxiv.org/abs/2606.25034
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Vision Encoder (simplified ViT-style) ────────────────────────────────────

class PatchEmbedding(nn.Module):
    """Simplified patch embedding for toy vision encoder."""
    def __init__(self, image_size: int = 224, patch_size: int = 16, embed_dim: int = 512):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W)
        B = x.size(0)
        x = self.proj(x)                          # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)          # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)            # (B, N+1, D)
        x = x + self.pos_embed
        return x


class VisualTransformerBlock(nn.Module):
    def __init__(self, embed_dim: int = 512, num_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y, _ = self.attn(y, y, y)
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x


class VisionEncoder(nn.Module):
    """Lightweight ViT-style vision encoder for safety features."""
    def __init__(self, embed_dim: int = 512, depth: int = 4, num_heads: int = 8):
        super().__init__()
        self.patch_embed = PatchEmbedding(embed_dim=embed_dim)
        self.blocks = nn.ModuleList([
            VisualTransformerBlock(embed_dim, num_heads)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, 0]  # return CLS token as global visual feature


# ─── Text Encoder (simplified) ────────────────────────────────────────────────

class TextEncoder(nn.Module):
    """Simplified text encoder with positional embeddings."""
    def __init__(self, vocab_size: int = 32000, embed_dim: int = 512,
                 max_len: int = 256, num_layers: int = 4, num_heads: int = 8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0)
        x = self.embed(input_ids) + self.pos_embed(positions)

        if attention_mask is not None:
            # Convert HuggingFace-style mask (1=attend, 0=ignore) to PyTorch convention
            src_key_padding_mask = attention_mask.eq(0)
        else:
            src_key_padding_mask = None

        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        x = self.norm(x)
        return x[:, 0]  # CLS-style pooling (first token)


# ─── Multimodal Fusion ────────────────────────────────────────────────────────

class MultimodalFusion(nn.Module):
    """Cross-modal attention fusion for vision + language."""
    def __init__(self, embed_dim: int = 512):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, visual: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        # Both are (B, D)
        v = visual.unsqueeze(1)   # (B, 1, D)
        t = text.unsqueeze(1)     # (B, 1, D)
        fused, _ = self.cross_attn(v, t, t)
        fused = self.norm(fused.squeeze(1))
        combined = torch.cat([visual, fused], dim=-1)
        return self.proj(combined)


# ─── YuvionVL Core Model ──────────────────────────────────────────────────────

class YuvionVLModel(nn.Module):
    """
    Simplified YuvionVL model for content safety classification.

    Architecture:
        VisionEncoder → visual features
        TextEncoder   → text features
        MultimodalFusion → fused representation
        SafetyHead    → violation type + confidence

    The full model (paper) uses a much larger LLM backbone with:
    - 8B/32B parameters
    - Full autoregressive text generation for reasoning
    - Chain-of-thought safety explanations
    - Three-stage training (pretraining → SFT → C2FT)
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_violation_types: int = 5,
        visual_depth: int = 4,
        text_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.vision_encoder = VisionEncoder(embed_dim=embed_dim, depth=visual_depth)
        self.text_encoder = TextEncoder(embed_dim=embed_dim, num_layers=text_layers)
        self.fusion = MultimodalFusion(embed_dim=embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Binary safety head (safe / violation)
        self.safety_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 2),
        )

        # Violation type head (multi-class, conditioned on flagged content)
        self.violation_type_head = nn.Linear(embed_dim, num_violation_types)

        # Reasoning projection (maps features to text space for CoT generation)
        # In paper: full LLM decoder generates the reasoning chain
        # Here: simplified projection for demonstration
        self.reasoning_proj = nn.Linear(embed_dim, embed_dim)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode text and optional image into a joint safety representation."""
        text_features = self.text_encoder(input_ids, attention_mask)

        if pixel_values is not None:
            visual_features = self.vision_encoder(pixel_values)
            features = self.fusion(visual_features, text_features)
        else:
            features = text_features

        return self.dropout(features)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        features = self.encode(input_ids, attention_mask, pixel_values)

        safety_logits = self.safety_head(features)      # (B, 2)
        type_logits = self.violation_type_head(features) # (B, num_types)

        output = {
            "safety_logits": safety_logits,
            "violation_type_logits": type_logits,
            "features": features,
        }

        if labels is not None:
            loss = F.cross_entropy(safety_logits, labels)
            output["loss"] = loss

        return output


# ─── C2FT: Confuse-then-Contrast Fine-Tuning ─────────────────────────────────

class C2FTLoss(nn.Module):
    """
    Confuse-then-Contrast Fine-Tuning loss (C2FT).

    Paper description:
    "C2FT mines model-specific confusions and performs multi-image joint
    contrastive supervision, substantially improving fine-grained safety
    discrimination on adversarial examples."

    Implementation:
    Given a batch of (anchor, confused) pairs where:
    - anchor: the correctly-labeled example the model was uncertain about
    - confused: the example the model confused with the anchor

    We apply InfoNCE-style contrastive loss in feature space, ensuring
    the model's representations become more discriminative for hard cases.

    In full paper:
    - Pairs are mined online from the model's own prediction failures
    - Multiple confused examples per anchor (multi-image joint supervision)
    - Combined with standard cross-entropy on the anchor
    """

    def __init__(self, temperature: float = 0.07, alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha   # weight for contrastive loss vs cross-entropy

    def forward(
        self,
        anchor_features: torch.Tensor,    # (B, D) — features of anchors
        confused_features: torch.Tensor,  # (B, D) — features of confused examples
        anchor_labels: torch.Tensor,      # (B,)   — true labels of anchors
        anchor_logits: torch.Tensor,      # (B, 2) — safety logits for anchors
    ) -> dict:
        B, D = anchor_features.shape

        # ── Standard cross-entropy on anchors ────────────────────────────────
        ce_loss = F.cross_entropy(anchor_logits, anchor_labels)

        # ── Contrastive loss (InfoNCE) ────────────────────────────────────────
        # Normalize features to unit sphere
        anc = F.normalize(anchor_features, dim=-1)       # (B, D)
        con = F.normalize(confused_features, dim=-1)     # (B, D)

        # Similarity matrix between anchors and confused examples
        # We want: anchor representations to be PULLED AWAY from confused examples
        # (since their true labels differ)
        # sim[i, j] = cosine similarity between anchor_i and confused_j
        sim = torch.mm(anc, con.T) / self.temperature    # (B, B)

        # In C2FT: the "positive" pairs are where anchor and confused have DIFFERENT labels
        # (i.e., we push them apart), and we use negatives as other items in the batch
        # Simplified: InfoNCE where diagonal pairs are hard negatives to push apart
        # Full C2FT: also includes same-label pairs from the confusion mining process

        # Diagonal: model-specific confusion pairs (push apart)
        labels_contrastive = torch.zeros(B, dtype=torch.long, device=anchor_features.device)
        # In InfoNCE framing, we want similarity to be lowest at the diagonal
        # We invert: use -sim so that minimizing cross-entropy pushes diagonal similarity down
        contrastive_loss = F.cross_entropy(-sim, labels_contrastive)

        # ── Combined loss ─────────────────────────────────────────────────────
        total_loss = (1 - self.alpha) * ce_loss + self.alpha * contrastive_loss

        return {
            "loss": total_loss,
            "ce_loss": ce_loss.item(),
            "contrastive_loss": contrastive_loss.item(),
        }


# ─── Safety Gate (inference) ──────────────────────────────────────────────────

class SafetyGate(nn.Module):
    """
    Inference-time safety gate for content policy enforcement.

    In production deployment: wraps the YuvionVL model and returns
    structured safety verdicts with violation types and confidence scores.
    """

    def __init__(self, model: YuvionVLModel, threshold: float = 0.5):
        super().__init__()
        self.model = model
        self.threshold = threshold

        self.violation_type_names = [
            "logo_counterfeit",
            "product_category_violation",
            "price_fraud",
            "illicit_goods",
            "ai_generated_fake",
        ]

    @torch.no_grad()
    def check(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
    ) -> list[dict]:
        output = self.model(input_ids, attention_mask, pixel_values)
        safety_probs = torch.softmax(output["safety_logits"], dim=-1)  # (B, 2)
        violation_probs = safety_probs[:, 1]   # probability of violation

        type_probs = torch.softmax(output["violation_type_logits"], dim=-1)
        top_types = type_probs.argmax(dim=-1)

        results = []
        for i in range(len(violation_probs)):
            is_violation = violation_probs[i].item() > self.threshold
            results.append({
                "is_violation": is_violation,
                "confidence": violation_probs[i].item(),
                "violation_type": self.violation_type_names[top_types[i].item()] if is_violation else "none",
                "type_confidence": type_probs[i, top_types[i]].item(),
            })

        return results
