# Valley3 — Code Reproduction

Faithful PyTorch reproduction of the core components from:

> **Valley3: Scaling Omni Foundation Models for E-commerce**  
> arXiv: 2605.01278 | ByteDance / Lark AI

## Structure

```
Valley3/
├── README.md
├── model/
│   ├── valley3.py          # Omni MLLM architecture
│   ├── audio_encoder.py    # Multilingual audio encoder
│   ├── vision_encoder.py   # Visual feature extraction
│   └── controllable_reasoning.py  # Reasoning mode controller
├── training/
│   ├── stage1_audio.py     # Stage 1: Audio understanding pre-training
│   ├── stage2_crossmodal.py # Stage 2: Cross-modal instruction tuning
│   ├── stage3_ecom.py      # Stage 3: E-commerce domain knowledge
│   └── stage4_longctx.py   # Stage 4: Long-context reasoning
├── agentic/
│   └── agentic_search.py   # Agentic search capability
├── data/
│   ├── toy_ecom_dataset.py # Toy e-commerce dataset
│   └── data_collator.py    # Multi-modal data collation
├── eval/
│   └── evaluate.py         # Evaluation on e-commerce benchmarks
└── train.py                # Main training script
```

## Quick Start

```bash
# Install dependencies
pip install torch transformers datasets einops

# Run toy training (Stage 1 only)
python train.py --stage 1 --epochs 1 --batch_size 2

# Evaluate
python eval/evaluate.py --checkpoint ./checkpoints/stage1
```
