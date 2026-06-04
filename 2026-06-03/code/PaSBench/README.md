# PaSBench-Video: Reproduction

Faithful PyTorch reproduction of the PaSBench-Video evaluation framework from:
> "PaSBench-Video: A Streaming Video Benchmark for Proactive Safety Warning" (arXiv:2606.02443)

## Structure

```
PaSBench/
├── data/
│   ├── dataset.py          # PaSBench-Video dataset loader
│   └── toy_data.py         # Toy dataset generator for testing
├── models/
│   ├── base_judge.py       # Abstract base class for video safety judges
│   └── mllm_judge.py       # MLLM-based safety judge wrapper
├── evaluation/
│   ├── metrics.py          # Temporal safety warning metrics (strict + lenient)
│   └── evaluator.py        # Main evaluation pipeline
├── benchmark.py            # Main benchmark runner
└── requirements.txt
```

## Usage

```bash
pip install -r requirements.txt
python benchmark.py --model seed_2_pro --split risk --metric strict
```
