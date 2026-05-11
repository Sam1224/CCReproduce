# Valley3: Scaling Omni Foundation Models for E-commerce — Code Reproduction

**Paper:** [arXiv:2605.01278](https://arxiv.org/abs/2605.01278)  
**Authors:** Zeyu Chen, Guanghao Zhou, et al. (AIDC-AI / Alibaba)  
**Score:** 84/100 ✦

## Overview

This folder contains a faithful PyTorch reproduction of Valley3's core architecture and training pipeline. The implementation follows the paper's described methodology:

1. **Omni MLLM Architecture** — Vision Encoder + Audio Encoder + MoE LLM backbone with cross-modal projectors
2. **Four-Stage Training** — Audio integration → Cross-modal instruction following → E-commerce domain → Long-context reasoning
3. **Controllable Reasoning Modes** — Non-thinking / Light-CoT / Heavy-CoT switching
4. **Agentic Search Stubs** — Tool-calling interface for e-commerce deep research
5. **E-commerce Evaluation** — 6-task benchmark evaluation script

## Files

```
Valley3/
├── README.md           ← This file
├── model/
│   ├── __init__.py
│   ├── vision_encoder.py     ← CLIP-based vision encoder stub
│   ├── audio_encoder.py      ← Whisper-based audio encoder stub
│   ├── moe_llm.py            ← MoE LLM backbone (simplified)
│   ├── valley3.py            ← Full Valley3 model assembly
│   └── projectors.py         ← Cross-modal projection layers
├── training/
│   ├── __init__.py
│   ├── stage1_audio.py       ← Stage 1: audio understanding pre-training
│   ├── stage2_crossmodal.py  ← Stage 2: cross-modal instruction following
│   ├── stage3_ecom.py        ← Stage 3: e-commerce domain knowledge
│   ├── stage4_longcontext.py ← Stage 4: long-context reasoning
│   └── dataset.py            ← Toy dataset (interface-aligned)
├── inference/
│   ├── __init__.py
│   ├── reasoning_modes.py    ← Controllable reasoning (No-Think/Light/Heavy)
│   └── agentic_search.py     ← Agentic search tool stubs
├── evaluation/
│   ├── __init__.py
│   └── ecom_benchmark.py     ← 6-task e-commerce evaluation
└── train.py                  ← End-to-end training entry point
```

## Quick Start

```bash
pip install torch transformers einops

# Run toy training pipeline (all 4 stages, small model)
python train.py --model_size tiny --stages 1,2,3,4 --toy_data

# Run evaluation on toy e-commerce benchmark
python -m evaluation.ecom_benchmark --model_path ./checkpoints/stage4

# Run inference with different reasoning modes
python inference/reasoning_modes.py --mode heavy --prompt "分析这个商品是否存在虚假宣传"
```
