"""
UNIVID: Unified Vision-Language Model for Video Moderation
Reproduction — model architecture

Paper: arXiv 2606.05748
Architecture: LLaVA-OV style (Visual Encoder + LLM backbone)
+ policy-aware caption generation head
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    CLIPVisionModel,
    CLIPImageProcessor,
)
from typing import Optional, List


class VisualEncoder(nn.Module):
    """Lightweight visual encoder wrapping CLIP-ViT.
    In the paper: InternViT or similar ViT backbone used.
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch16"):
        super().__init__()
        self.clip = CLIPVisionModel.from_pretrained(model_name)
        self.hidden_size = self.clip.config.hidden_size

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, T, C, H, W) — video frames
        Returns:
            visual_features: (B, T*num_patches, hidden_size)
        """
        B, T, C, H, W = pixel_values.shape
        frames = pixel_values.view(B * T, C, H, W)
        outputs = self.clip(pixel_values=frames)
        # (B*T, num_patches, hidden_size)
        patch_features = outputs.last_hidden_state[:, 1:, :]  # drop [CLS]
        num_patches = patch_features.size(1)
        return patch_features.view(B, T * num_patches, self.hidden_size)


class ProjectionLayer(nn.Module):
    """MLP projection: visual_hidden_size → llm_hidden_size"""

    def __init__(self, visual_dim: int, llm_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(visual_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class UNIVID(nn.Module):
    """
    UNIVID: Unified VLM for Video Moderation.

    Generates policy-aware captions for videos given moderation policy text.
    The caption embedding serves as a shared intermediate representation for:
      1. Risk Filter (embedding similarity)
      2. Moderation Actor (classification)
      3. Trend Governance (clustering)

    Training recipe (from paper §3):
      Phase 1: SFT on general video-captioning data
      Phase 2: Instruction-tune on policy-specific moderation data
               (hybrid: expert annotations + synthetic data)
    """

    def __init__(
        self,
        llm_name: str = "Qwen/Qwen2-1.5B",  # toy; paper uses larger backbone
        visual_encoder_name: str = "openai/clip-vit-base-patch16",
        max_caption_length: int = 256,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name, trust_remote_code=True)
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_name, trust_remote_code=True
        )
        self.visual_encoder = VisualEncoder(visual_encoder_name)
        self.projection = ProjectionLayer(
            visual_dim=self.visual_encoder.hidden_size,
            llm_dim=self.llm.config.hidden_size,
        )
        self.max_caption_length = max_caption_length

    def encode_video(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode video frames into projected visual tokens."""
        visual_features = self.visual_encoder(pixel_values)
        return self.projection(visual_features)

    def forward(
        self,
        pixel_values: torch.Tensor,
        policy_input_ids: torch.Tensor,
        caption_input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass for policy-aware caption generation.

        Args:
            pixel_values: (B, T, C, H, W) video frames
            policy_input_ids: (B, L_policy) tokenised policy text
            caption_input_ids: (B, L_cap) tokenised target caption (training)
            labels: (B, L_cap) caption labels for LM loss

        Returns:
            CausalLMOutput with loss (training) or logits (inference)

        Paper formula (§4.1):
            caption = argmax_C P(C | V, Policy; θ_UNIVID)
        """
        visual_tokens = self.encode_video(pixel_values)  # (B, T*P, D)
        policy_embeds = self.llm.get_input_embeddings()(policy_input_ids)  # (B, L, D)

        # Concatenate: [visual tokens | policy tokens | caption tokens]
        inputs_embeds = torch.cat([visual_tokens, policy_embeds], dim=1)
        if caption_input_ids is not None:
            cap_embeds = self.llm.get_input_embeddings()(caption_input_ids)
            inputs_embeds = torch.cat([inputs_embeds, cap_embeds], dim=1)

        # Build attention mask
        B = pixel_values.size(0)
        vis_len = visual_tokens.size(1)
        pol_len = policy_embeds.size(1)
        cap_len = caption_input_ids.size(1) if caption_input_ids is not None else 0
        total_len = vis_len + pol_len + cap_len
        attn_mask = torch.ones(B, total_len, device=pixel_values.device)

        # Shift labels to align with caption portion only
        if labels is not None:
            pad_labels = torch.full(
                (B, vis_len + pol_len), -100, dtype=torch.long, device=labels.device
            )
            labels = torch.cat([pad_labels, labels], dim=1)

        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            labels=labels,
        )

    @torch.no_grad()
    def get_caption_embedding(
        self, pixel_values: torch.Tensor, policy_input_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate caption and return its mean-pooled hidden-state embedding.
        Used by Risk Filter and Trend Governance modules.

        Returns:
            embedding: (B, D) — pooled representation
        """
        visual_tokens = self.encode_video(pixel_values)
        policy_embeds = self.llm.get_input_embeddings()(policy_input_ids)
        inputs_embeds = torch.cat([visual_tokens, policy_embeds], dim=1)
        attn_mask = torch.ones(
            inputs_embeds.size(0), inputs_embeds.size(1), device=inputs_embeds.device
        )
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            output_hidden_states=True,
        )
        last_hidden = outputs.hidden_states[-1]  # (B, L, D)
        embedding = last_hidden.mean(dim=1)  # mean pool
        return embedding

    @torch.no_grad()
    def generate_caption(
        self,
        pixel_values: torch.Tensor,
        policy_input_ids: torch.Tensor,
        max_new_tokens: int = 128,
    ) -> List[str]:
        """Autoregressively generate policy-aware captions."""
        visual_tokens = self.encode_video(pixel_values)
        policy_embeds = self.llm.get_input_embeddings()(policy_input_ids)
        inputs_embeds = torch.cat([visual_tokens, policy_embeds], dim=1)
        attn_mask = torch.ones(
            inputs_embeds.size(0), inputs_embeds.size(1), device=inputs_embeds.device
        )
        output_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        captions = [
            self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids
        ]
        return captions


class ModerationClassificationHead(nn.Module):
    """
    Moderation Actor — lightweight binary/multi-class head on top of
    UNIVID caption embeddings. Paper has two variants:
      - Precision Actor: high precision, low recall
      - Recall Actor: high recall, moderate precision
    """

    def __init__(self, hidden_dim: int, num_violation_types: int = 10):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_violation_types),
        )

    def forward(self, caption_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            caption_embedding: (B, D)
        Returns:
            logits: (B, num_violation_types)
        """
        return self.classifier(caption_embedding)
