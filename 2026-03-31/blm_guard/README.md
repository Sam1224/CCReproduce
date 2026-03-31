# BLM-Guard (reproduction-oriented implementation)

Reference paper: **BLM-Guard: Explainable Multimodal Ad Moderation with Chain-of-Thought and Policy-Aligned Rewards** (arXiv:2602.18193, AAAI 2026)

This folder provides a **runnable, non-trivial** PyTorch implementation that mirrors the paper’s pipeline structure:

- **Risk-prompt anchored keyframe selection** (BIN+TOP) over a sequence of frame features.
- **Patch/region saliency selection** (implemented as differentiable patch-gating over synthetic patch features).
- **Multimodal encoder** (Transformer) over video + ASR/OCR text.
- **Explainable CoT decoder** (Transformer decoder) that generates `<think> ... </think> <answer> ... </answer>`.
- **Policy-aligned RL fine-tuning** using a GRPO-style group advantage update with:
  - correctness reward (scene/type/risky)
  - format reward (`<think>` then `<answer>`)
  - consistency reward (rationale mentions predicted label tokens)
  - KL regularization to a frozen reference policy

## Quickstart

```bash
cd ccreproduce_repo/2026-03-31/blm_guard
python3 -m pip install -r requirements.txt
python3 train.py --epochs 3 --rl_steps 200 --rollouts 8
python3 test.py --ckpt checkpoints/blm_guard.pt
```

## Data

By default, the scripts run on a **synthetic but structured** dataset.

If you have real moderation logs, you can provide a JSONL file:

```json
{"frames": [[...]], "asr": ["..."], "ocr": ["..."], "scene": 1, "ad_type": 2, "risky": 1, "rationale": "<think>...</think><answer>...</answer>"}
```

and run:

```bash
python3 train.py --data path/to/data.jsonl
```

## Explicit pseudocode (not implementable without real VLM / proprietary data)

### (A) Frame extraction from raw videos

```text
# NOT IMPLEMENTED
# decode video -> sample clips -> run vision backbone -> frame embeddings
frames = VideoDecoder(video_path).sample(fps=1)
frame_feats = VisionBackbone(frames)
```

### (B) Large-LLM CoT supervision distillation

```text
# NOT IMPLEMENTED
# teacher LLM generates high-quality CoT and policy-aligned preferences
cot = TeacherLLM(prompt + frames + asr + ocr)
preference = PreferenceModel(cot)
# student learns from (cot, preference)
```
