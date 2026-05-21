# RuleSafe-VL — Code Reproduction

Faithful PyTorch reproduction of the benchmark framework from:

> **RuleSafe-VL: Evaluating Rule-Conditioned Decision Reasoning in Vision-Language Content Moderation**  
> Zhifeng Lu, Dianyun Wang, Yuhu Shang, Zhenbo Xu  
> Beijing University of Posts and Telecommunications, May 2026  
> arXiv: [2605.07760](https://arxiv.org/abs/2605.07760)

---

## Files

```
code/RuleSafe-VL/
├── README.md              ← This file
├── benchmark/
│   ├── rule_graph.py      ← Rule graph (93 atomic rules + 92 typed relations)
│   ├── case_format.py     ← Case schema (image + text + rule set + labels)
│   ├── toy_dataset.py     ← Toy dataset generator (interface-aligned)
│   └── metrics.py         ← Rule-retrieval / rule-application / final-decision metrics
├── evaluation/
│   ├── evaluator.py       ← Main evaluation loop
│   ├── chain_evaluator.py ← Full rule-chain accuracy evaluator
│   └── prompts.py         ← Standard evaluation prompts for VLMs
├── models/
│   └── vlm_wrapper.py     ← Thin wrapper around HuggingFace VLMs (LLaVA, InternVL, etc.)
└── run_eval.py            ← Entry point: python run_eval.py --model llava --cases data/toy
```

## Quick Start

```bash
pip install torch transformers pillow
python run_eval.py --model llava-1.6 --cases data/toy --output results/
```
