# Taiji — Code Reproduction

**Paper:** Taiji: Pareto Optimal Policy Optimization with Semantics-IDs Trade-off for Industrial LLM-Enhanced Recommendation  
**arXiv:** https://arxiv.org/abs/2606.03866  
**Institution:** Kuaishou Technology  

## Structure

```
code/Taiji/
├── README.md
├── requirements.txt
├── data/
│   └── toy_dataset.py          # Toy dataset generator
├── model/
│   ├── llm_enhancer.py         # LLM-as-Enhancer backbone
│   ├── popo.py                 # Pareto Optimal Policy Optimization (POPO)
│   └── cot_data_gen.py         # Reverse-Engineered CoT data generation
├── train.py                    # SFT + RL training pipeline
└── evaluate.py                 # Offline evaluation
```

## Quick Start

```bash
pip install -r requirements.txt
python data/toy_dataset.py     # Generate toy data
python train.py --stage sft    # Stage 1: SFT with CoT data
python train.py --stage rl     # Stage 2: RL with POPO
python evaluate.py             # Evaluate on toy dataset
```
