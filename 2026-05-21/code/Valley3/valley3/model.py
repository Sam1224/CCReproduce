"""Valley3: Omni-modal MLLM for e-commerce (toy reproduction of arXiv:2605.01278).

Architecture overview:
  - ImageEncoder   : lightweight patch CNN (→ image tokens)
  - VideoEncoder   : frame-level ImageEncoder + temporal pooling (→ video tokens)
  - AudioEncoder   : CNN on log-mel spectrogram (→ audio tokens)
  - TextEmbedder   : token embedding + positional embedding
  - ModalityProjector : projects each modality to LLM hidden dim
  - OmniLLM        : lightweight Transformer LLM backbone
  - Task heads     : classification, VQA, moderation, captioning, audio_cls

Four-stage training in paper (simplified to single-stage toy):
  Stage 1: Audio alignment (audio encoder + text)
  Stage 2: Cross-modal instruction following (all modalities)
  Stage 3: E-commerce domain knowledge injection
  Stage 4: Long-context reasoning

In this toy we implement the architecture and a unified training objective.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .dataset import NUM_CLASSES


# ---------------------------------------------------------------------------
# Modality encoders
# ---------------------------------------------------------------------------

class PatchCNN(nn.Module):
    """(B, C, H, W) -> (B, Np, d_model)"""

    def __init__(self, in_ch: int = 3, d_model: int = 64, num_patches: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, d_model, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(d_model, d_model, 3, stride=2, padding=1), nn.GELU(),
        )
        self.num_patches = num_patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)                           # (B, d, H', W')
        B, d, H, W = h.shape
        h = h.flatten(2).transpose(1, 2)              # (B, H'W', d)
        if h.size(1) != self.num_patches:
            h = F.adaptive_avg_pool1d(
                h.transpose(1, 2), self.num_patches
            ).transpose(1, 2)
        return h


class VideoEncoder(nn.Module):
    """(B, T, C, H, W) -> (B, Np, d_model)"""

    def __init__(self, d_model: int = 64, num_patches: int = 16):
        super().__init__()
        self.frame_enc = PatchCNN(d_model=d_model, num_patches=num_patches)
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)  # pool over frames

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = video.shape
        frames = video.flatten(0, 1)               # (BT, C, H, W)
        tokens = self.frame_enc(frames)            # (BT, Np, d)
        tokens = tokens.view(B, T, tokens.size(1), tokens.size(2))
        # Mean pool over time
        tokens = tokens.mean(dim=1)               # (B, Np, d)
        return tokens


class AudioEncoder(nn.Module):
    """(B, F, T) log-mel -> (B, Na, d_model)"""

    def __init__(self, d_model: int = 64, num_audio_tokens: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, (3, 3), padding=1), nn.GELU(),
            nn.Conv2d(16, d_model, (3, 3), stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.num_audio_tokens = num_audio_tokens
        self.proj = nn.Linear(d_model * 16, d_model * num_audio_tokens)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """audio: (B, F, T) -> tokens: (B, Na, d_model)"""
        x = audio.unsqueeze(1)              # (B, 1, F, T)
        h = self.encoder(x)                 # (B, d, 4, 4)
        B, d, _, _ = h.shape
        h = h.flatten(1)                    # (B, d*16)
        tokens = self.proj(h)               # (B, d*Na)
        d_model = tokens.size(-1) // self.num_audio_tokens
        return tokens.view(B, self.num_audio_tokens, d_model)


class TextEmbedder(nn.Module):
    """(B, L) -> (B, L, d_model)"""

    def __init__(self, vocab_size: int = 512, d_model: int = 64, max_len: int = 128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = nn.Embedding(max_len, d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, L = tokens.shape
        pos = torch.arange(L, device=tokens.device).unsqueeze(0)
        return self.emb(tokens) + self.pos(pos)


# ---------------------------------------------------------------------------
# Modality projector (aligns each modality to LLM hidden dim)
# ---------------------------------------------------------------------------

class ModalityProjector(nn.Module):
    def __init__(self, in_dim: int = 64, out_dim: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.GELU(), nn.LayerNorm(out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ---------------------------------------------------------------------------
# Lightweight LLM backbone (Transformer)
# ---------------------------------------------------------------------------

class OmniLLM(nn.Module):
    """Small transformer backbone that processes concatenated modal tokens."""

    def __init__(self, d_model: int = 128, nhead: int = 4, num_layers: int = 4,
                 ffn_dim: int = 256):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model, nhead, ffn_dim, batch_first=True, norm_first=True
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Full Valley3 model
# ---------------------------------------------------------------------------

class Valley3Model(nn.Module):
    """Valley3: Omni-modal LLM for e-commerce.

    Input  : any combination of {text, image, video, audio}
    Output : task-specific prediction (cls logits / caption tokens)

    Four-stage training (paper):
        Stage 1 — audio alignment: only AudioEncoder + TextEmbedder trained
        Stage 2 — cross-modal SFT: all encoders + LLM trained with instruction data
        Stage 3 — e-commerce domain: fine-tune on ecom-specific data
        Stage 4 — long-context: adapt to multi-turn/long-video data

    In this toy we implement a single unified stage that jointly handles all modalities.
    """

    def __init__(
        self,
        vocab_size: int = 512,
        d_enc: int = 64,
        d_llm: int = 128,
        num_img_patches: int = 16,
        num_audio_tokens: int = 8,
        text_len: int = 32,
        num_llm_layers: int = 4,
    ):
        super().__init__()
        # Encoders
        self.img_enc = PatchCNN(d_model=d_enc, num_patches=num_img_patches)
        self.vid_enc = VideoEncoder(d_model=d_enc, num_patches=num_img_patches)
        self.aud_enc = AudioEncoder(d_model=d_enc, num_audio_tokens=num_audio_tokens)
        self.txt_emb = TextEmbedder(vocab_size=vocab_size, d_model=d_enc, max_len=text_len)

        # Projectors (align each modality to LLM dim)
        self.img_proj = ModalityProjector(d_enc, d_llm)
        self.vid_proj = ModalityProjector(d_enc, d_llm)
        self.aud_proj = ModalityProjector(d_enc, d_llm)
        self.txt_proj = ModalityProjector(d_enc, d_llm)

        # LLM backbone
        self.llm = OmniLLM(d_model=d_llm, num_layers=num_llm_layers)

        # Task-specific heads
        self.cls_head = nn.Linear(d_llm, NUM_CLASSES["classification"])
        self.vqa_head = nn.Linear(d_llm, NUM_CLASSES["vqa"])
        self.mod_head = nn.Linear(d_llm, 2)          # binary violation detection
        self.cap_head = nn.Linear(d_llm, vocab_size)  # token logits
        self.aud_head = nn.Linear(d_llm, NUM_CLASSES["audio_cls"])

        self.d_llm = d_llm

        # Stage 1 flag: only train audio and text (for paper's Stage-1 alignment)
        self._stage = "full"

    def set_stage(self, stage: str):
        """Control which parameters are active (mirrors 4-stage training)."""
        assert stage in ("audio_align", "cross_modal", "ecom_domain", "long_context", "full")
        self._stage = stage
        if stage == "audio_align":
            for p in self.parameters():
                p.requires_grad = False
            for p in self.aud_enc.parameters():
                p.requires_grad = True
            for p in self.aud_proj.parameters():
                p.requires_grad = True
        else:
            for p in self.parameters():
                p.requires_grad = True

    def forward(
        self,
        task: str,
        text: torch.Tensor,
        image: torch.Tensor,
        video: torch.Tensor,
        audio: torch.Tensor,
    ) -> torch.Tensor:
        """Forward for a single task.

        task   : task name string (all must be the same within a batch for toy)
        text   : (B, L)
        image  : (B, C, H, W)
        video  : (B, T, C, H, W)
        audio  : (B, F, T_a)
        Returns: logits or caption logits depending on task
        """
        # Encode and project each modality
        txt_tok = self.txt_proj(self.txt_emb(text))          # (B, L, d_llm)
        img_tok = self.img_proj(self.img_enc(image))          # (B, Np, d_llm)
        vid_tok = self.vid_proj(self.vid_enc(video))          # (B, Np, d_llm)
        aud_tok = self.aud_proj(self.aud_enc(audio))          # (B, Na, d_llm)

        # Task-routing: choose which modalities to concatenate
        # (In paper, routing is determined by instruction prompt; here we use task name)
        if isinstance(task, (list, tuple)):
            task_name = task[0] if task else "classification"
        else:
            task_name = task

        if task_name in ("classification", "vqa", "moderation"):
            tokens = torch.cat([txt_tok, img_tok], dim=1)
        elif task_name == "captioning":
            tokens = torch.cat([img_tok, txt_tok[:, :1]], dim=1)
        elif task_name == "audio_cls":
            tokens = torch.cat([aud_tok, txt_tok[:, :4]], dim=1)
        else:
            tokens = torch.cat([txt_tok, img_tok, vid_tok, aud_tok], dim=1)

        # LLM processing
        hidden = self.llm(tokens)                    # (B, seq_len, d_llm)
        pooled = hidden.mean(dim=1)                  # (B, d_llm)

        # Task head
        if task_name == "classification":
            return self.cls_head(pooled)
        elif task_name == "vqa":
            return self.vqa_head(pooled)
        elif task_name == "moderation":
            return self.mod_head(pooled)
        elif task_name == "captioning":
            return self.cap_head(hidden)              # (B, seq, vocab)
        elif task_name == "audio_cls":
            return self.aud_head(pooled)
        else:
            return self.cls_head(pooled)
