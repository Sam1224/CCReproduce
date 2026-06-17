# DynamicLiveModeration — Reproduction Code

Paper: "Dynamic Content Moderation in Livestreams: Combining Supervised Classification with MLLM-Boosted Similarity Matching"  
arXiv: https://arxiv.org/abs/2512.03553  
Venue: ACM KDD 2026

## Structure

```
DynamicLiveModeration/
├── README.md
├── requirements.txt
├── data/
│   └── toy_dataset.py          # Toy multimodal livestream dataset
├── models/
│   ├── mllm_teacher.py         # MLLM teacher (knowledge distillation source)
│   ├── classification_pipeline.py   # Supervised classification pipeline
│   └── similarity_pipeline.py       # Reference-based similarity pipeline
├── train/
│   ├── train_classifier.py     # Train classification pipeline
│   └── train_similarity.py     # Train similarity pipeline with KD
├── eval/
│   └── evaluate.py             # Evaluate at target precision/recall
└── inference/
    └── dual_pipeline_inference.py   # Production dual-pipeline inference
```

## Quick Start

```bash
pip install -r requirements.txt
python data/toy_dataset.py          # Generate toy data
python train/train_classifier.py    # Train classification head
python train/train_similarity.py    # Train similarity model with KD
python eval/evaluate.py             # Evaluate precision/recall
python inference/dual_pipeline_inference.py  # Run dual pipeline
```
