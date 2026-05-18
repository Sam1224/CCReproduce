# EVADE — Code Reproduction

Reproduction of: **EVADE: Multimodal Benchmark for Evasive Content Detection in E-Commerce Applications**  
ArXiv: https://arxiv.org/abs/2505.17654

## Structure

```
EVADE/
├── README.md              # This file
├── requirements.txt       # Dependencies
├── data/
│   ├── dataset.py         # Dataset loading & interface
│   └── toy_data.py        # Toy dataset generator (for testing)
├── eval/
│   ├── evaluator.py       # Core evaluation framework
│   ├── metrics.py         # F1, Accuracy, Partial-match metrics
│   └── prompts.py         # Prompt templates (Single-Violation & All-in-One)
├── models/
│   └── model_interface.py # Unified interface for LLM/VLM backends
└── run_eval.py            # Main evaluation script
```

## Quick Start

```bash
pip install -r requirements.txt

# Run evaluation with toy data (no API key needed)
python run_eval.py --mode single_violation --model dummy

# Run with a real model (requires API key)
export OPENAI_API_KEY=<your_key>
python run_eval.py --mode all_in_one --model gpt-4o --data_path data/toy_data
```

## Notes

- The original EVADE dataset is not publicly released; this reproduction uses a toy dataset with the same interface.
- Policy rules are included as prompt templates based on the paper's description.
