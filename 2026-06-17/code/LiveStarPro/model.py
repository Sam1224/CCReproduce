"""
LiveStarPro — Model Architecture

Three core components (faithful to the paper):
  1. SCAM  (Streaming Causal Attention Masks)   — training strategy
  2. TSHM  (Tree-Structured Hierarchical Memory) — long-horizon memory
  3. SVeD  (Streaming Verification Decoding)     — proactive response timing

Paper formula references:
  TSHM compression: m_t = Compress(m_{t-1}, v_evicted)   (Eq. 1)
  SVeD score:       pplx_t = -log P(response | frames_t)  (Eq. 2)
  Response trigger: respond iff pplx_t < θ                (Eq. 3)
  SCAM mask:        M_{ij} = 1 iff j ≤ i (causal + stream position)  (Eq. 4)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Tuple


# ── SCAM: Streaming Causal Attention Mask ──────────────────────────────────

def build_scam_mask(
    seq_len: int,
    chunk_size: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Build Streaming Causal Attention Mask (SCAM).
    Within each chunk: full causal attention.
    Between chunks: only the last frame of previous chunk is visible.
    (Eq. 4: enforces incremental video-language alignment)
    """
    mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)
    for i in range(seq_len):
        # Can attend to: all frames in same chunk, and boundary frames of prior chunks
        chunk_start = (i // chunk_size) * chunk_size
        # Causal: all positions ≤ i in same chunk
        for j in range(chunk_start, i + 1):
            mask[i, j] = True
        # Cross-chunk: only the last frame of each prior chunk
        for c_start in range(0, chunk_start, chunk_size):
            c_end = c_start + chunk_size - 1
            if c_end < seq_len:
                mask[i, c_end] = True
    return ~mask  # True = masked out (nn.MultiheadAttention convention)


# ── TSHM: Tree-Structured Hierarchical Memory ─────────────────────────────

class TreeMemoryNode:
    """A node in the Tree-Structured Hierarchical Memory."""
    def __init__(self, embedding: torch.Tensor, level: int = 0):
        self.embedding = embedding   # (D,)
        self.level = level
        self.children: List["TreeMemoryNode"] = []
        self.event_summary: Optional[str] = None


class TreeStructuredHierarchicalMemory(nn.Module):
    """
    TSHM: Recursively organizes evicted KV cache frames into event chains.
    When KV cache capacity is exceeded, frames are compressed into a tree node.

    Compression: new_node = MLP(mean(evicted_frames))   (Eq. 1)
    Retrieval: query attends over tree nodes at each level
    """

    def __init__(self, embed_dim: int = 256, max_children: int = 4, max_levels: int = 3):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_children = max_children
        self.max_levels = max_levels

        # Compression MLP (summarizes a group of frame embeddings into one node)
        self.compress_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # Retrieval attention (query over tree nodes)
        self.retrieval_attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)

    def compress(self, frame_embs: torch.Tensor) -> torch.Tensor:
        """
        Compress a group of frame embeddings into one memory node embedding.
        Args:
            frame_embs: (N, D)
        Returns:
            node_emb: (D,)
        """
        avg = frame_embs.mean(dim=0, keepdim=True)  # (1, D)
        return self.compress_mlp(avg).squeeze(0)      # (D,)

    def retrieve(
        self,
        query: torch.Tensor,          # (B, 1, D) — current query embedding
        memory_nodes: torch.Tensor,   # (B, M, D) — flattened tree nodes
    ) -> torch.Tensor:
        """
        Retrieve relevant memory using attention over tree nodes.
        Returns: (B, D) augmented context
        """
        out, _ = self.retrieval_attn(query, memory_nodes, memory_nodes)
        return out.squeeze(1)  # (B, D)

    def build_tree_from_frames(
        self, frame_embs: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Incrementally build tree from a list of frame embeddings.
        Returns flattened list of node embeddings for retrieval.
        """
        if len(frame_embs) == 0:
            return []

        frame_embs_tensor = torch.stack(frame_embs)  # (N, D)
        nodes = []
        chunk = self.max_children

        # Level 0: leaf nodes (groups of max_children frames)
        for i in range(0, len(frame_embs), chunk):
            group = frame_embs_tensor[i:i + chunk]
            node = self.compress(group)
            nodes.append(node)

        # Higher levels: compress groups of level-0 nodes
        current_level = nodes
        for level in range(1, self.max_levels):
            if len(current_level) <= 1:
                break
            next_level = []
            for i in range(0, len(current_level), chunk):
                group = torch.stack(current_level[i:i + chunk])
                node = self.compress(group)
                next_level.append(node)
            nodes.extend(next_level)
            current_level = next_level

        return nodes


# ── SVeD: Streaming Verification Decoding ─────────────────────────────────

class StreamingVerificationDecoding(nn.Module):
    """
    SVeD: Decides WHEN to respond by computing a perplexity-based score
    on a candidate response prefix for each incoming frame.

    respond iff perplexity(response | frames_t) < θ    (Eq. 2, 3)

    In practice: a lightweight verifier MLP replaces full perplexity computation.
    """

    def __init__(self, embed_dim: int = 256, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold
        # Lightweight verifier (approximates perplexity as a learned score)
        self.verifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, frame_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            frame_emb: (B, D)
        Returns:
            respond_score: (B,) in [0, 1]  — 1 = should respond
            should_respond: (B,) bool mask
        """
        score = self.verifier(frame_emb).squeeze(-1)   # (B,) — Eq. 2 approximation
        should_respond = score > self.threshold         # Eq. 3
        return score, should_respond


# ── Frame & Video Encoder (Vision Encoder simplified) ─────────────────────

class FrameEncoder(nn.Module):
    """
    Encodes a single video frame into a feature embedding.
    In the real paper: CLIP/SigLIP visual encoder.
    Here: simple CNN stem as proxy.
    """

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, 4),   # very lightweight for toy
            nn.GELU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.proj = nn.Linear(32 * 4 * 4, embed_dim)

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frame: (B, 3, H, W)
        Returns:
            emb: (B, embed_dim)
        """
        x = self.conv(frame)
        x = x.flatten(1)
        return self.proj(x)


# ── Full LiveStarPro Model ─────────────────────────────────────────────────

class LiveStarPro(nn.Module):
    """
    Full LiveStarPro model combining SCAM training, TSHM memory, SVeD decoding.

    During training:
      - Frames are processed in chunks (SCAM masked attention)
      - Response prediction is trained as classification (did/when to respond)

    During inference (streaming):
      - TSHM stores compressed history beyond KV cache
      - SVeD determines response timing
    """

    def __init__(
        self,
        embed_dim: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        chunk_size: int = 8,         # frames per streaming chunk
        kv_cache_size: int = 32,     # max frames in KV cache
        memory_max_children: int = 4,
        sved_threshold: float = 0.5,
        n_response_classes: int = 10, # toy: number of response categories
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.chunk_size = chunk_size
        self.kv_cache_size = kv_cache_size

        # Frame encoder (vision backbone)
        self.frame_encoder = FrameEncoder(embed_dim)

        # Temporal Transformer with SCAM masks (Eq. 4)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 4,
            batch_first=True, dropout=0.1, norm_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Tree-Structured Hierarchical Memory (Eq. 1)
        self.tshm = TreeStructuredHierarchicalMemory(embed_dim, memory_max_children)

        # Streaming Verification Decoding (Eq. 2, 3)
        self.sved = StreamingVerificationDecoding(embed_dim, sved_threshold)

        # Response head
        self.response_head = nn.Linear(embed_dim, n_response_classes)

        # Memory-augmented context projector
        self.ctx_proj = nn.Linear(embed_dim * 2, embed_dim)

    def forward_training(
        self,
        frames: torch.Tensor,  # (B, T, 3, H, W) — T frames
        respond_labels: torch.Tensor,  # (B,) — frame index to respond at
        response_labels: torch.Tensor, # (B,) — response category
    ) -> dict:
        """
        Training forward with SCAM masks.
        """
        B, T, C, H, W = frames.shape

        # Encode all frames
        frame_embs = self.frame_encoder(frames.view(B * T, C, H, W))  # (B*T, D)
        frame_embs = frame_embs.view(B, T, -1)                         # (B, T, D)

        # Build SCAM mask
        scam_mask = build_scam_mask(T, self.chunk_size, frames.device)  # (T, T)
        scam_mask = scam_mask.unsqueeze(0).expand(B * self.temporal_transformer.layers[0].self_attn.num_heads, -1, -1)

        # Temporal modeling (Eq. 4)
        ctx = self.temporal_transformer(frame_embs)  # (B, T, D)

        # Compute SVeD score per frame (Eq. 2)
        sved_scores, _ = self.sved(ctx.view(B * T, -1))
        sved_scores = sved_scores.view(B, T)                             # (B, T)

        # SVeD timing loss: BCE on whether each frame is the respond frame
        timing_targets = torch.zeros(B, T, device=frames.device)
        for b in range(B):
            if respond_labels[b] < T:
                timing_targets[b, respond_labels[b]] = 1.0
        L_timing = F.binary_cross_entropy(sved_scores, timing_targets)

        # Response loss at the respond frame
        respond_ctx = ctx[torch.arange(B), respond_labels.clamp(0, T - 1)]  # (B, D)
        response_logits = self.response_head(respond_ctx)
        L_response = F.cross_entropy(response_logits, response_labels)

        loss = L_response + L_timing
        return {
            "loss": loss,
            "L_response": L_response.item(),
            "L_timing": L_timing.item(),
            "sved_scores": sved_scores,
        }

    @torch.no_grad()
    def stream_inference(
        self,
        frame: torch.Tensor,          # (B, 3, H, W) — single incoming frame
        kv_cache: List[torch.Tensor], # list of past frame embeddings in cache
        memory_nodes: List[torch.Tensor],  # flattened TSHM node embeddings
    ) -> dict:
        """
        Online inference for a single incoming frame.
        Returns whether to respond and the response if so.
        """
        # Encode new frame
        frame_emb = self.frame_encoder(frame)   # (B, D)

        # Evict oldest frame if cache full (TSHM: compress evicted)
        if len(kv_cache) >= self.kv_cache_size:
            evicted = kv_cache.pop(0)
            # Compress evicted frame into TSHM (Eq. 1)
            # (in real: compress a group of evicted frames)
            node_emb = self.tshm.compress(evicted.unsqueeze(0))
            memory_nodes.append(node_emb)

        kv_cache.append(frame_emb)

        # Retrieve relevant memory context
        if memory_nodes:
            mem_tensor = torch.stack(memory_nodes).unsqueeze(0).expand(frame_emb.shape[0], -1, -1)
            mem_ctx = self.tshm.retrieve(frame_emb.unsqueeze(1), mem_tensor)
            frame_emb_aug = self.ctx_proj(torch.cat([frame_emb, mem_ctx], dim=-1))
        else:
            frame_emb_aug = frame_emb

        # SVeD: determine if we should respond (Eq. 2, 3)
        score, should_respond = self.sved(frame_emb_aug)

        result = {"score": score, "should_respond": should_respond}
        if should_respond.any():
            response_logits = self.response_head(frame_emb_aug)
            result["response"] = response_logits.argmax(dim=-1)

        return result


if __name__ == "__main__":
    model = LiveStarPro(
        embed_dim=128,
        n_heads=4,
        n_layers=2,
        chunk_size=4,
        kv_cache_size=16,
        n_response_classes=5,
    )

    # Training test
    B, T, H, W = 2, 16, 32, 32
    frames = torch.randn(B, T, 3, H, W)
    respond_labels = torch.randint(0, T, (B,))
    response_labels = torch.randint(0, 5, (B,))

    out = model.forward_training(frames, respond_labels, response_labels)
    print(f"Training loss: {out['loss'].item():.4f}")
    print(f"L_response: {out['L_response']:.4f}, L_timing: {out['L_timing']:.4f}")

    # Streaming inference test
    kv_cache = []
    memory_nodes = []
    frame = torch.randn(1, 3, H, W)
    for i in range(20):
        result = model.stream_inference(frame, kv_cache, memory_nodes)
        if result["should_respond"].any():
            print(f"Frame {i}: RESPOND (score={result['score'].item():.3f}, class={result['response'].item()})")
        else:
            print(f"Frame {i}: silent (score={result['score'].item():.3f})")

    print("Model OK.")
