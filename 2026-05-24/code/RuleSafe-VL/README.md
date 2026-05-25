# RuleSafe-VL — Reproduction

Faithful PyTorch reproduction of the benchmark evaluation framework described in:

> **RuleSafe-VL: Evaluating Rule-Conditioned Decision Reasoning in Vision-Language Content Moderation**  
> Zhifeng Lu, Dianyuan Wang, Yuhu Shang, Zhenbo Xu  
> arXiv: 2605.07760 (May 2026)

## Structure

```
RuleSafe-VL/
├── README.md
├── data/
│   ├── rules.json           # Example rule graph (93 atomic rules, 92 relations)
│   ├── sample_cases.json    # 20 toy image-text moderation cases
│   └── generate_toy_data.py # Script to generate synthetic benchmark data
├── models/
│   ├── rule_graph.py        # Rule formalism: atomic rules + relation types
│   ├── vlm_evaluator.py     # Wrapper for VLM inference (OpenAI / HuggingFace)
│   └── rule_conditioned_judge.py  # Rule-conditioned decision reasoning module
├── eval/
│   ├── metrics.py           # Evaluation metrics: rule activation F1, interaction accuracy
│   └── benchmark.py         # Main benchmark runner
├── train/
│   └── finetune_vlm.py      # Optional: fine-tune VLM on rule-conditioned data
└── main.py                  # Entry point
```

## Requirements

```
torch>=2.0
transformers>=4.40
openai>=1.30
Pillow>=10.0
networkx>=3.0
numpy>=1.24
scikit-learn>=1.3
tqdm>=4.65
```

## Quick Start

```bash
pip install -r requirements.txt

# Generate toy dataset
python data/generate_toy_data.py --output data/

# Run benchmark evaluation with a local VLM
python main.py \
    --model_type huggingface \
    --model_name Qwen/Qwen2-VL-7B-Instruct \
    --rules_path data/rules.json \
    --cases_path data/sample_cases.json \
    --output_dir results/

# Run benchmark with OpenAI GPT-4V
python main.py \
    --model_type openai \
    --model_name gpt-4o \
    --rules_path data/rules.json \
    --cases_path data/sample_cases.json \
    --output_dir results/
```
