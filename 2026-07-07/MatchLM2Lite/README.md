# MatchLM2Lite — Code Reproduction

Toy-but-runnable reproduction of **MatchLM2Lite: A Scalable MLLM-to-Lite Framework for Reproduced Content Identification** (arXiv:2606.14786).

## File Structure

```
MatchLM2Lite/
├── data.py         # RCIDataset, VideoPair, simulate_visual/audio_features
├── model.py        # MatchLM (teacher), MatchLite (student), TrimodalFusion
├── train.py        # Teacher training + student distillation
├── eval.py         # F1, precision, recall; breakdown by reproduction type
└── README.md
```

## Quick Start

```bash
# Train teacher then distill student
python train.py --mode both --epochs 5 --distill_weight 0.5

# Evaluate with trained checkpoints
python eval.py --model both \
    --teacher_ckpt checkpoints/matchlm_teacher.pt \
    --student_ckpt checkpoints/matchlite_student.pt

# Teacher only
python train.py --mode teacher --epochs 5
python eval.py --model teacher --teacher_ckpt checkpoints/matchlm_teacher.pt
```

## Architecture Summary

```
Reference Video:   [visual frames] [audio segments] [title+desc+captions]
                         ↓               ↓                  ↓
                  VideoEncoder    AudioEncoder         TextEncoder
                         ↓               ↓                  ↓
                         └───────── TrimodalFusion ─────────┘
                                         ↓
                                   ref_embedding (D)

Candidate Video:   (same encoding) → cand_embedding (D)

Pairwise features: [ref, cand, ref*cand, |ref-cand|]  → size 4D
                         ↓
                     Scorer MLP → reproduction_score ∈ [0, 1]
```

## Knowledge Distillation (MatchLite)

```
MatchLM Teacher                   MatchLite Student
─────────────────                 ──────────────────────────────
VideoEncoder (4-layer)            video_enc: Linear→GELU→LN
AudioEncoder (2-layer)            audio_enc: Linear→GELU→LN
TextEncoder  (4-layer)            text_enc:  Embedding→mean-pool
TrimodalFusion (cross-attn)       fusion_proj: cat→Linear→GELU→LN
embed_dim = 512                   embed_dim = 256
                                  align_proj: 256 → 512
                                  (aligns to teacher space for distill)
```

**Distillation loss:**
```
L_distill = 0.5 × [MSE(align(ref_lite), teacher_ref.detach())
                 +  MSE(align(cand_lite), teacher_cand.detach())]
L_total = (1 - w) × L_label + w × L_distill
```

## Key Paper Claims (reproduced in spirit)

| Claim | Paper | This Reproduction |
|-------|-------|------------------|
| Three-modal encoding | Video + audio + text | Simulated V/A features + char-level text |
| MLLM as representation extractor | Not next-token prediction | Shared encoder → pairwise scoring |
| 35× inference cost reduction | Deployment cost | Parameter ratio ~10× (toy scale) |
| Fine-grained reproduction types | clip_repost, dubbed, caption_replaced, reframed | All four types in toy dataset |
| Knowledge distillation | Embedding + label supervision | MSE alignment + BCE label loss |

## Reproduction Types in Toy Dataset

| Type | Label | Description |
|------|-------|-------------|
| `repost_minor_edit` | 0.95 | Near-duplicate repost with minor edits |
| `dubbed` | 0.88 | Original video + replaced audio/voiceover |
| `clip_repost` | 0.92 | Clipped segment extracted from original |
| `caption_replaced` | 0.85 | Same video content, different captions |
| _(none)_ | 0.0 | Genuinely distinct content |

## Notes

- Production reproduction uses ViT/CLIP for visual features and wav2vec2 for audio.
- This reproduction simulates those features with `torch.randn` to demonstrate the model architecture.
- The character-level tokenizer is a toy approximation; production uses a BPE tokenizer.
- For production: integrate a pre-extracted feature pipeline and scale to millions of video pairs.
