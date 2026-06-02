# CS-VAR — Code Reproduction

**Paper:** Deja Vu in Plots: Leveraging Cross-Session Evidence with Retrieval-Augmented LLMs for Live Streaming Risk Assessment  
**arXiv:** https://arxiv.org/abs/2601.16027  
**Authors:** Yiran Qiao, Xiang Ao, Jing Chen, Yang Liu, Qiwei Zhong, Qing He (CAS)

## Architecture

```
Input: Current Live Stream Session
         │
  Session Encoder (lightweight)
         │
  ┌──────┴────────────────────┐
  │                           │
  │   Cross-Session Retrieval │
  │   (from history database) │
  │          │                │
  │   Retrieved Sessions      │
  │          │                │
  │     LLM Teacher           │
  │   (RAG + joint reasoning) │
  │          │                │
  │   Soft Labels / KD Signal │
  └──────────┤                │
             │ Knowledge      │
             │ Distillation   │
             ▼                │
      Student Risk Model      │
      (real-time inference)   │
             │                │
         Risk Score           │
```

## Files

- `model.py` — Session encoder + student risk model
- `retrieval.py` — Cross-session retrieval (faiss-based approximate NN)
- `llm_teacher.py` — LLM teacher reasoning over retrieved evidence
- `dataset.py` — Toy session dataset
- `train.py` — Training with KD
- `evaluate.py` — Risk assessment evaluation

## Quick Start

```bash
pip install torch numpy faiss-cpu
python train.py --epochs 10
python evaluate.py --checkpoint checkpoints/best.pt
```
