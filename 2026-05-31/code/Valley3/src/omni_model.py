"""
Omni MLLM for E-commerce (Valley3 toy implementation).

Real Valley3 architecture (paper Section 3.1):
  Qwen3-VL backbone + Audio Transformer + Audio MLP connector
  Unified token sequence: [text tokens] + [visual tokens] + [audio tokens]
  → LLM Backbone
  → Controllable reasoning (think level 0-3)
  → Agentic search (tool calls)

This toy replaces Qwen3-VL with a small Transformer LM for demonstration.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .audio_encoder import AudioTransformerEncoder, AudioMLPConnector


class ToyVisionEncoder(nn.Module):
    """Stub vision encoder (replaces CLIP/SigLIP in real Valley3)."""
    def __init__(self, img_dim: int = 64, output_dim: int = 512, num_patches: int = 16):
        super().__init__()
        self.num_patches = num_patches
        self.proj = nn.Linear(img_dim, output_dim)

    def forward(self, img_feat: torch.Tensor) -> torch.Tensor:
        """img_feat: [B, num_patches, img_dim] → [B, num_patches, output_dim]"""
        return self.proj(img_feat)


class ToyLMBackbone(nn.Module):
    """Minimal autoregressive Transformer backbone (replaces Qwen3-VL LM)."""
    def __init__(self, vocab_size: int = 256, d_model: int = 512,
                 nhead: int = 8, num_layers: int = 4, max_seq_len: int = 512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.d_model = d_model

    def forward(self, input_tokens: torch.Tensor,
                context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_tokens: [B, T_dec] integer token IDs
            context: [B, T_ctx, d_model] — multimodal context (visual + audio + text prefix)
        Returns:
            logits: [B, T_dec, vocab_size]
        """
        B, T = input_tokens.shape
        x = self.embed(input_tokens)
        pos = torch.arange(T, device=input_tokens.device)
        x = x + self.pos_emb(pos).unsqueeze(0)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        x = self.transformer(x, context, tgt_mask=causal_mask)
        return self.lm_head(x)


class OmniEcommerceModel(nn.Module):
    """
    Valley3 Omni E-commerce model (toy).
    Combines vision, audio, and text into a unified LLM input.
    """
    def __init__(self, vocab_size: int = 256, d_model: int = 512,
                 audio_mel_bins: int = 80, img_dim: int = 64,
                 num_audio_tokens: int = 8, num_visual_patches: int = 16):
        super().__init__()
        self.audio_enc = AudioTransformerEncoder(mel_bins=audio_mel_bins, output_dim=d_model)
        self.audio_connector = AudioMLPConnector(audio_dim=d_model, llm_dim=d_model,
                                                  num_tokens=num_audio_tokens)
        self.vision_enc = ToyVisionEncoder(img_dim=img_dim, output_dim=d_model,
                                           num_patches=num_visual_patches)
        self.lm = ToyLMBackbone(vocab_size=vocab_size, d_model=d_model)

    def build_context(self, text_embed: torch.Tensor,
                      img_feat: torch.Tensor = None,
                      mel: torch.Tensor = None) -> torch.Tensor:
        """
        Build unified context from text + visual + audio tokens.
        Paper: "audio embeddings are concatenated with visual and text tokens
                into a unified input space" (Section 3.1)
        """
        parts = [text_embed]  # [B, T_text, d_model]
        if img_feat is not None:
            vis_tokens = self.vision_enc(img_feat)    # [B, num_patches, d_model]
            parts.append(vis_tokens)
        if mel is not None:
            audio_emb = self.audio_enc(mel)           # [B, d_model]
            audio_tokens = self.audio_connector(audio_emb)  # [B, num_audio_tokens, d_model]
            parts.append(audio_tokens)
        return torch.cat(parts, dim=1)                # [B, T_ctx, d_model]

    def forward(self, input_ids: torch.Tensor,
                text_embed: torch.Tensor,
                img_feat: torch.Tensor = None,
                mel: torch.Tensor = None) -> torch.Tensor:
        context = self.build_context(text_embed, img_feat, mel)
        return self.lm(input_ids, context)
