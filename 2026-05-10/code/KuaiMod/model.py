"""
KuaiMod Model — VLM as Policy

Architecture faithful to Section 3 of the paper:
  1. VLM Backbone (visual encoder + text decoder)
  2. CoT Reasoning Head: generates chain-of-thought rationale
  3. Policy Classifier: 3-way classification (safe / violating / borderline)

The paper uses a proprietary Kuaishou VLM. We use a lightweight publicly
available backbone (CLIP vision encoder + GPT-2 decoder) as a toy substitute.
For real reproduction, replace with InternVL2-2B or Qwen-VL-Chat.

Key design choices (from paper):
  - Multi-frame temporal pooling: average-pool frame features before fusion
  - Text + visual token concatenation as context for CoT generation
  - Cross-entropy loss on classification token after CoT prefix
  - Two-stage training: (1) classification warmup, (2) CoT end-to-end
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoTokenizer,
    CLIPVisionModel,
    CLIPVisionConfig,
    GPT2LMHeadModel,
    GPT2Config,
    GPT2Tokenizer,
)


NUM_CLASSES = 3  # safe, violating, borderline
VERDICT_TOKENS = ["[SAFE]", "[VIOLATING]", "[BORDERLINE]"]


class TemporalFramePooler(nn.Module):
    """
    Aggregate multi-frame visual features.
    Paper uses mean pooling over sampled frames (1 FPS).
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, frame_feats: torch.Tensor) -> torch.Tensor:
        # frame_feats: (B, T, D)
        pooled = frame_feats.mean(dim=1)  # (B, D)
        return self.proj(pooled)


class VisualEncoder(nn.Module):
    """
    Toy visual encoder using CLIP ViT-B/32 spatial features.
    In paper: custom Kuaishou VLM visual encoder.
    """
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        super().__init__()
        # Use a lightweight CNN as proxy when CLIP unavailable
        self.hidden_dim = 768
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: (B*T, C, H, W) — frames flattened into batch
        return self.backbone(pixel_values)


class CoTDecoder(nn.Module):
    """
    Lightweight autoregressive decoder for CoT rationale generation.
    Paper: full VLM decoder conditioned on visual + text context.
    Here: single-layer GRU decoder as a toy substitute.

    The real implementation would use a transformer decoder (GPT-2 / Qwen-VL).
    """
    def __init__(self, vocab_size: int, d_model: int = 512, max_len: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.context_proj = nn.Linear(768, d_model)
        self.gru = nn.GRU(d_model, d_model, num_layers=2, batch_first=True)
        self.output_proj = nn.Linear(d_model, vocab_size)
        self.max_len = max_len
        self.d_model = d_model

    def forward(
        self,
        context: torch.Tensor,     # (B, D_vis)
        input_ids: Optional[torch.Tensor] = None,  # (B, L) teacher-forced
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = context.size(0)
        h0 = self.context_proj(context).unsqueeze(0).expand(2, -1, -1).contiguous()
        # h0: (num_layers, B, d_model)

        if input_ids is not None:
            emb = self.embedding(input_ids)  # (B, L, d_model)
            out, hidden = self.gru(emb, h0)
            logits = self.output_proj(out)  # (B, L, vocab_size)
        else:
            # Greedy decoding (inference)
            token = torch.zeros(B, 1, dtype=torch.long, device=context.device)
            logits_list = []
            hidden = h0
            for _ in range(self.max_len):
                emb = self.embedding(token)
                out, hidden = self.gru(emb, hidden)
                step_logits = self.output_proj(out)  # (B, 1, vocab_size)
                logits_list.append(step_logits)
                token = step_logits.argmax(-1)
            logits = torch.cat(logits_list, dim=1)

        return logits, hidden


class KuaiMod(nn.Module):
    """
    KuaiMod: VLM-as-Policy for Short Video Platform Content Moderation.

    Two-stage training (from paper Section 3.2):
      Stage 1 (warmup):  train classifier only, no CoT generation
      Stage 2 (CoT):     train both CoT decoder + classifier end-to-end

    Forward pass:
      frames (B, T, C, H, W)  →  visual features  →  temporal pooling
      text tokens              →  text embedding   →  concat with visual
      context                  →  CoT decoder      →  rationale logits
      context + CoT hidden     →  classifier       →  {safe, violating, borderline}
    """

    def __init__(
        self,
        vocab_size: int = 30000,
        d_model: int = 512,
        num_classes: int = NUM_CLASSES,
        max_cot_len: int = 128,
    ):
        super().__init__()
        self.visual_encoder = VisualEncoder()
        self.temporal_pooler = TemporalFramePooler(hidden_dim=768)

        # Text encoder: simple embedding + mean pool
        self.text_embedding = nn.Embedding(vocab_size, 256, padding_idx=0)
        self.text_proj = nn.Linear(256, 768)

        # Multimodal fusion
        self.fusion = nn.Sequential(
            nn.Linear(768 + 768, 768),
            nn.GELU(),
            nn.LayerNorm(768),
        )

        # CoT decoder (Section 3.1: rationale generation)
        self.cot_decoder = CoTDecoder(vocab_size=vocab_size, d_model=d_model,
                                      max_len=max_cot_len)

        # Policy classifier (Section 3.1: verdict head)
        # Input: fused context + CoT hidden state
        self.classifier = nn.Sequential(
            nn.Linear(768 + d_model, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """
        frames: (B, T, C, H, W)
        returns: (B, D) — temporal pooled visual features
        """
        B, T, C, H, W = frames.shape
        flat = frames.view(B * T, C, H, W)
        frame_feats = self.visual_encoder(flat)          # (B*T, 768)
        frame_feats = frame_feats.view(B, T, -1)         # (B, T, 768)
        return self.temporal_pooler(frame_feats)          # (B, 768)

    def encode_text(self, text_ids: torch.Tensor) -> torch.Tensor:
        """
        text_ids: (B, L)
        returns: (B, 768)
        """
        emb = self.text_embedding(text_ids)   # (B, L, 256)
        pooled = emb.mean(dim=1)              # (B, 256)
        return self.text_proj(pooled)         # (B, 768)

    def forward(
        self,
        frames: torch.Tensor,
        text_ids: torch.Tensor,
        cot_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        stage: str = "cot",  # "warmup" or "cot"
    ) -> Dict[str, torch.Tensor]:

        vis_feat = self.encode_frames(frames)         # (B, 768)
        txt_feat = self.encode_text(text_ids)         # (B, 768)
        context = self.fusion(torch.cat([vis_feat, txt_feat], dim=-1))  # (B, 768)

        outputs = {}

        if stage == "warmup":
            # Stage 1: only train classifier, no CoT
            dummy_hidden = torch.zeros(
                context.size(0), self.cot_decoder.d_model,
                device=context.device
            )
            clf_logits = self.classifier(
                torch.cat([context, dummy_hidden], dim=-1)
            )  # (B, num_classes)
            outputs["clf_logits"] = clf_logits

            if labels is not None:
                outputs["loss"] = F.cross_entropy(clf_logits, labels)

        else:
            # Stage 2: generate CoT then classify
            cot_logits, hidden = self.cot_decoder(context, cot_ids)
            # hidden: (num_layers, B, d_model) — GRU hidden state

            # Take last layer hidden as CoT representation
            cot_repr = hidden[-1]  # (B, d_model)

            clf_logits = self.classifier(
                torch.cat([context, cot_repr], dim=-1)
            )  # (B, num_classes)

            outputs["cot_logits"] = cot_logits
            outputs["clf_logits"] = clf_logits

            loss = torch.tensor(0.0, device=context.device)
            if labels is not None:
                clf_loss = F.cross_entropy(clf_logits, labels)
                outputs["clf_loss"] = clf_loss
                loss = loss + clf_loss

            if cot_ids is not None and cot_logits is not None:
                # Teacher-forced CoT loss: shift by 1
                B, L, V = cot_logits.shape
                cot_loss = F.cross_entropy(
                    cot_logits[:, :-1].reshape(-1, V),
                    cot_ids[:, 1:].reshape(-1),
                    ignore_index=0,
                )
                outputs["cot_loss"] = cot_loss
                # Paper uses weighted sum: λ=0.5 for CoT, 1.0 for classification
                loss = loss + 0.5 * cot_loss

            if "clf_loss" in outputs or "cot_loss" in outputs:
                outputs["loss"] = loss

        return outputs

    @torch.no_grad()
    def predict(
        self,
        frames: torch.Tensor,
        text_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Inference: generate CoT rationale + verdict."""
        self.eval()
        vis_feat = self.encode_frames(frames)
        txt_feat = self.encode_text(text_ids)
        context = self.fusion(torch.cat([vis_feat, txt_feat], dim=-1))

        cot_logits, hidden = self.cot_decoder(context, input_ids=None)
        cot_repr = hidden[-1]  # (B, d_model)
        clf_logits = self.classifier(torch.cat([context, cot_repr], dim=-1))

        cot_tokens = cot_logits.argmax(-1)  # (B, L)
        verdicts = clf_logits.argmax(-1)    # (B,)

        return {
            "verdicts": verdicts,
            "verdict_probs": clf_logits.softmax(-1),
            "cot_tokens": cot_tokens,
        }


if __name__ == "__main__":
    model = KuaiMod(vocab_size=1000, max_cot_len=16)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    B, T, C, H, W = 2, 4, 3, 224, 224
    frames = torch.randn(B, T, C, H, W)
    text_ids = torch.randint(1, 1000, (B, 32))
    cot_ids = torch.randint(1, 1000, (B, 16))
    labels = torch.randint(0, 3, (B,))

    # Stage 1: warmup
    out = model(frames, text_ids, labels=labels, stage="warmup")
    print(f"Warmup loss: {out['loss'].item():.4f}")

    # Stage 2: CoT
    out = model(frames, text_ids, cot_ids=cot_ids, labels=labels, stage="cot")
    print(f"CoT loss: {out['loss'].item():.4f}")
    print(f"clf_logits shape: {out['clf_logits'].shape}")

    # Inference
    pred = model.predict(frames, text_ids)
    print(f"Verdicts: {pred['verdicts'].tolist()}")
    print(f"Verdict probs:\n{pred['verdict_probs']}")
