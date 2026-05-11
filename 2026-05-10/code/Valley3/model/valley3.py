"""
Valley3: Scaling Omni Foundation Models for E-commerce
Reproduction of arXiv:2605.01278

Main model assembly: Vision + Audio + MoE LLM + Cross-modal projectors.

Architecture (paper Section 3):
  - Vision Encoder: CLIP-ViT variant, processes images/video frames
  - Audio Encoder: Whisper-based, processes speech/audio in short-video e-commerce
  - MoE LLM: Mixture-of-Experts language model backbone
  - V-Projector: Maps vision tokens to LLM embedding space
  - A-Projector: Maps audio tokens to LLM embedding space

Four modalities: text, image, video, audio → unified token sequence → MoE LLM
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Tuple


class Valley3Config:
    """Valley3 model configuration."""

    def __init__(
        self,
        # Vision encoder
        vision_model_name: str = "openai/clip-vit-large-patch14-336",
        vision_hidden_size: int = 1024,
        # Audio encoder
        audio_model_name: str = "openai/whisper-medium",
        audio_hidden_size: int = 1024,
        # LLM backbone
        llm_hidden_size: int = 4096,
        llm_num_layers: int = 32,
        llm_num_heads: int = 32,
        llm_vocab_size: int = 128256,  # Llama-3-style vocab
        # MoE settings (paper uses MoE LLM)
        num_experts: int = 8,
        num_experts_per_token: int = 2,
        # Projector
        projector_hidden_size: int = 4096,
        # E-commerce specific
        max_num_frames: int = 32,   # max video frames
        max_audio_tokens: int = 256,  # max audio tokens per segment
        max_seq_length: int = 32768,
        # Reasoning mode
        reasoning_mode: str = "non_thinking",  # non_thinking / light / heavy
    ):
        self.vision_model_name = vision_model_name
        self.vision_hidden_size = vision_hidden_size
        self.audio_model_name = audio_model_name
        self.audio_hidden_size = audio_hidden_size
        self.llm_hidden_size = llm_hidden_size
        self.llm_num_layers = llm_num_layers
        self.llm_num_heads = llm_num_heads
        self.llm_vocab_size = llm_vocab_size
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.projector_hidden_size = projector_hidden_size
        self.max_num_frames = max_num_frames
        self.max_audio_tokens = max_audio_tokens
        self.max_seq_length = max_seq_length
        self.reasoning_mode = reasoning_mode


class VisionProjector(nn.Module):
    """
    Cross-modal projector for vision tokens → LLM embedding space.
    Paper uses MLP projector; we implement a two-layer MLP with GELU.
    """

    def __init__(self, vision_hidden_size: int, llm_hidden_size: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(vision_hidden_size, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        # vision_features: [batch, num_tokens, vision_hidden_size]
        return self.proj(vision_features)


class AudioProjector(nn.Module):
    """
    Cross-modal projector for audio tokens → LLM embedding space.
    Valley3's key contribution: native multilingual audio for short-video e-commerce.
    """

    def __init__(self, audio_hidden_size: int, llm_hidden_size: int):
        super().__init__()
        # Audio tokens are typically longer; we use a compression layer
        # kernel_size=1 to work with any sequence length (including stub single-token)
        self.compress = nn.Conv1d(
            audio_hidden_size, audio_hidden_size, kernel_size=1, stride=1
        )
        self.proj = nn.Sequential(
            nn.Linear(audio_hidden_size, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        # audio_features: [batch, audio_hidden_size, seq_len]
        compressed = self.compress(audio_features)   # [batch, audio_hidden, seq_len]
        # [batch, seq_len, audio_hidden_size]
        compressed = compressed.transpose(1, 2)
        return self.proj(compressed)


class SimpleMoELayer(nn.Module):
    """
    Simplified Mixture-of-Experts FFN layer.
    Paper uses MoE LLM; we provide a minimal faithful implementation.

    Formula (standard MoE):
      y = sum_{i in TopK(G(x))} G_i(x) * Expert_i(x)
    where G(x) = Softmax(TopK(W_g @ x))
    """

    def __init__(self, hidden_size: int, num_experts: int, num_experts_per_token: int):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        ffn_dim = hidden_size * 4

        # Router
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

        # Expert FFNs (SwiGLU-style like LLaMA)
        self.w1 = nn.ModuleList([nn.Linear(hidden_size, ffn_dim, bias=False) for _ in range(num_experts)])
        self.w2 = nn.ModuleList([nn.Linear(ffn_dim, hidden_size, bias=False) for _ in range(num_experts)])
        self.w3 = nn.ModuleList([nn.Linear(hidden_size, ffn_dim, bias=False) for _ in range(num_experts)])
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, hidden = x.shape
        x_flat = x.view(-1, hidden)  # [B*T, H]

        # Router logits and top-k selection
        router_logits = self.gate(x_flat)  # [B*T, num_experts]
        topk_weights, topk_indices = torch.topk(
            router_logits, self.num_experts_per_token, dim=-1
        )
        topk_weights = torch.softmax(topk_weights, dim=-1)  # normalize

        # Dispatch to experts
        output = torch.zeros_like(x_flat)
        for k in range(self.num_experts_per_token):
            expert_idx = topk_indices[:, k]  # [B*T]
            weight = topk_weights[:, k : k + 1]  # [B*T, 1]
            for e in range(self.num_experts):
                mask = (expert_idx == e)
                if mask.any():
                    x_e = x_flat[mask]
                    # SwiGLU FFN
                    out_e = self.w2[e](self.act(self.w1[e](x_e)) * self.w3[e](x_e))
                    output[mask] += weight[mask] * out_e

        return output.view(batch, seq, hidden)


class Valley3Attention(nn.Module):
    """Multi-head attention with RoPE (simplified)."""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, H = hidden_states.shape
        q = self.q_proj(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** -0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = torch.softmax(attn_weights, dim=-1)
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, H)
        return self.o_proj(out)


class Valley3TransformerLayer(nn.Module):
    """Single transformer layer with MoE FFN (paper architecture)."""

    def __init__(self, config: Valley3Config):
        super().__init__()
        self.attn = Valley3Attention(config.llm_hidden_size, config.llm_num_heads)
        self.moe = SimpleMoELayer(
            config.llm_hidden_size, config.num_experts, config.num_experts_per_token
        )
        self.norm1 = nn.RMSNorm(config.llm_hidden_size)
        self.norm2 = nn.RMSNorm(config.llm_hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-norm attention
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states

        # Pre-norm MoE FFN
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.moe(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Valley3Model(nn.Module):
    """
    Valley3 full model: Vision + Audio + MoE LLM.

    Input modalities:
      - text: token IDs (always present)
      - image/video frames: pixel values → vision encoder → V-projector
      - audio: mel spectrogram → audio encoder → A-projector

    All modality tokens are concatenated and fed to the MoE LLM.
    Special tokens: <img_start>, <img_end>, <vid_start>, <vid_end>,
                    <aud_start>, <aud_end>
    """

    def __init__(self, config: Valley3Config):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embedding = nn.Embedding(config.llm_vocab_size, config.llm_hidden_size)

        # Vision encoder stub (replace with real CLIP/SigLIP in production)
        # Use adaptive pooling to handle arbitrary image sizes
        self._vision_pool_size = 8  # pool to 8x8 spatial tokens
        self.vision_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(self._vision_pool_size),
            nn.Flatten(),
            nn.Linear(3 * self._vision_pool_size * self._vision_pool_size, config.vision_hidden_size),
            nn.GELU(),
            nn.Linear(config.vision_hidden_size, config.vision_hidden_size),
        )
        self.vision_projector = VisionProjector(
            config.vision_hidden_size, config.llm_hidden_size
        )

        # Audio encoder stub (replace with real Whisper encoder in production)
        # Use adaptive pooling to handle arbitrary mel spectrogram lengths
        self._audio_pool_size = 32  # pool to fixed time dimension
        self.audio_encoder = nn.Sequential(
            nn.AdaptiveAvgPool1d(self._audio_pool_size),
            nn.Flatten(),
            nn.Linear(80 * self._audio_pool_size, config.audio_hidden_size),
            nn.GELU(),
            nn.Linear(config.audio_hidden_size, config.audio_hidden_size),
        )
        self.audio_projector = AudioProjector(
            config.audio_hidden_size, config.llm_hidden_size
        )

        # MoE LLM backbone
        self.layers = nn.ModuleList(
            [Valley3TransformerLayer(config) for _ in range(config.llm_num_layers)]
        )
        self.norm = nn.RMSNorm(config.llm_hidden_size)

        # Language model head
        self.lm_head = nn.Linear(config.llm_hidden_size, config.llm_vocab_size, bias=False)

        # Tie embeddings (standard practice)
        self.lm_head.weight = self.token_embedding.weight

    def encode_vision(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Encode image/video frames.
        pixel_values: [B, num_frames, C, H, W] or [B, C, H, W] for single image
        Returns: [B, T, llm_hidden_size] — one token per frame per sample
        """
        if pixel_values.dim() == 5:
            B, T, C, H, W = pixel_values.shape
            frames = pixel_values.view(B * T, C, H, W)
        else:
            B, C, H, W = pixel_values.shape
            T = 1
            frames = pixel_values  # [B, C, H, W]

        # vision_encoder: AdaptiveAvgPool2d → Flatten → Linear → GELU → Linear
        vision_feats = self.vision_encoder(frames)  # [B*T, vision_hidden]
        vision_feats = vision_feats.unsqueeze(1)     # [B*T, 1, vision_hidden]
        vision_tokens = self.vision_projector(vision_feats)  # [B*T, 1, llm_hidden]
        # Reshape back to [B, T, llm_hidden] for concatenation
        return vision_tokens.view(B, T, self.config.llm_hidden_size)

    def encode_audio(self, mel_spectrograms: torch.Tensor) -> torch.Tensor:
        """
        Encode audio mel spectrograms.
        mel_spectrograms: [B, mel_bins, time_frames] e.g. [B, 80, 3000]
        Returns: [B, compressed_tokens, llm_hidden_size]
        """
        B, mel_bins, time = mel_spectrograms.shape
        # audio_encoder: AdaptiveAvgPool1d → Flatten → Linear → GELU → Linear
        audio_feats = self.audio_encoder(mel_spectrograms)  # [B, audio_hidden]
        # Expand to sequence (stub: treat as single token)
        audio_feats = audio_feats.unsqueeze(1)     # [B, 1, audio_hidden]
        # Transpose for Conv1d in projector: [B, audio_hidden, 1]
        audio_feats_t = audio_feats.transpose(1, 2)
        audio_tokens = self.audio_projector(audio_feats_t)  # [B, 1, llm_hidden]
        return audio_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        mel_spectrograms: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        reasoning_mode: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for Valley3.

        Args:
            input_ids: [B, seq_len] text token IDs (with placeholder tokens for
                       image/audio positions)
            pixel_values: [B, num_frames, C, H, W] or None
            mel_spectrograms: [B, mel_bins, time_frames] or None
            attention_mask: [B, seq_len] causal mask
            labels: [B, seq_len] for language modeling loss
            reasoning_mode: override config.reasoning_mode at inference time
        """
        # 1. Text token embeddings
        hidden_states = self.token_embedding(input_ids)  # [B, T, llm_hidden]

        # 2. Inject vision tokens (in-place replacement at <img> positions)
        if pixel_values is not None:
            vision_tokens = self.encode_vision(pixel_values)
            # In full implementation: replace placeholder token positions
            # Simplified: prepend to sequence
            hidden_states = torch.cat([vision_tokens, hidden_states], dim=1)

        # 3. Inject audio tokens
        if mel_spectrograms is not None:
            audio_tokens = self.encode_audio(mel_spectrograms)
            hidden_states = torch.cat([audio_tokens, hidden_states], dim=1)

        # 4. Apply reasoning mode prefix (paper: special system prompt tokens)
        mode = reasoning_mode or self.config.reasoning_mode
        # In full implementation: prepend thinking-mode control tokens here
        # See inference/reasoning_modes.py for detailed implementation

        # 5. MoE LLM forward
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        hidden_states = self.norm(hidden_states)

        # 6. LM head
        logits = self.lm_head(hidden_states)  # [B, T, vocab_size]

        # 7. Compute loss if labels provided
        loss = None
        if labels is not None:
            # Prepended modality tokens have no labels; pad with -100 to match
            # logit sequence length before applying next-token-prediction shift.
            T_logits = logits.shape[1]
            T_labels = labels.shape[1]
            if T_logits > T_labels:
                # Pad labels at the front (for prepended modality tokens)
                pad = torch.full(
                    (labels.shape[0], T_logits - T_labels), -100,
                    dtype=labels.dtype, device=labels.device
                )
                labels = torch.cat([pad, labels], dim=1)
            elif T_labels > T_logits:
                labels = labels[:, :T_logits]

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, self.config.llm_vocab_size),
                shift_labels.view(-1),
            )

        return {"loss": loss, "logits": logits}


def build_valley3_tiny() -> Valley3Model:
    """Build a tiny Valley3 for testing (fits on single GPU/CPU)."""
    config = Valley3Config(
        vision_hidden_size=256,
        audio_hidden_size=256,
        llm_hidden_size=512,
        llm_num_layers=4,
        llm_num_heads=8,
        llm_vocab_size=32000,
        num_experts=4,
        num_experts_per_token=2,
    )
    return Valley3Model(config)


if __name__ == "__main__":
    import torch

    print("Building Valley3 (tiny) for smoke test...")
    model = build_valley3_tiny()
    model.eval()

    batch_size = 2
    seq_len = 16

    # Toy inputs
    input_ids = torch.randint(0, 32000, (batch_size, seq_len))
    # Tiny image: 3x336x336 → but we use tiny config, so just [B, 1, C, H, W]
    pixel_values = torch.randn(batch_size, 1, 3, 336, 336)
    # Audio: [B, 80, 3000] mel spectrogram
    mel_spec = torch.randn(batch_size, 80, 3000)
    labels = torch.randint(0, 32000, (batch_size, seq_len + 2))  # +2 for prepended tokens

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            mel_spectrograms=mel_spec,
            labels=labels,
        )

    print(f"  Logits shape: {out['logits'].shape}")
    print(f"  Loss: {out['loss'].item():.4f}")
    print("Smoke test PASSED.")
