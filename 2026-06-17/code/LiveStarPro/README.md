# LiveStarPro: Proactive Streaming Video Understanding

Reproduction of **"LiveStarPro: Proactive Streaming Video Understanding with Hierarchical Memory for Long-Horizon Streams"** (arXiv 2606.17798).

## Quick Start

```bash
pip install -r requirements.txt
python train.py --epochs 20
python test.py --ckpt liveStarPro_best.pt --demo
```

## Architecture

```
Incoming Video Stream
    │
    ▼ (frame-by-frame)
FrameEncoder (vision backbone)
    │
    ├──[KV Cache full?]──→  TSHM.compress(evicted frames) → Memory Tree Node
    │                                                              │
    ▼                                                              ▼
TemporalTransformer (with SCAM mask)           TSHM.retrieve(query, memory_nodes)
    │                                                              │
    └──────────────────────── ctx_proj ──────────────────────────┘
                                    │
                              SVeD Verifier
                                    │
                   ┌────────────────┴────────────────┐
                silent (score < θ)          RESPOND (score ≥ θ)
                                                      │
                                              ResponseHead → class
```

## Three Core Components (Paper Eqs. 1-4)

| Component | Role | Paper Eq. |
|-----------|------|-----------|
| **SCAM** | Streaming causal attention masks for incremental training | Eq. 4 |
| **TSHM** | Tree-structured hierarchical memory compressing evicted KV frames | Eq. 1 |
| **SVeD** | Perplexity-based proactive response timing | Eq. 2, 3 |

## Paper Results (vs. Prior Online Video-LLMs)
- Semantic Correctness: **+28.9%**
- Timing Error: **-18.2%**
- Inference Speedup: **1.58×** (streaming KV cache)

## Key Simplifications
- Toy random frames (real: actual video streams)
- Lightweight CNN (real: CLIP/SigLIP visual encoder)
- SVeD verifier = learned MLP (real: LLM perplexity computed in single forward pass)
