# MORE — Toy Reproduction

Paper: One Model, Multiple Goals: Adaptive Multi-Objective Learning for E-commerce Dialogue Systems  
arXiv: https://arxiv.org/abs/2606.09293

This folder implements a CPU-friendly PyTorch version of MORE's core idea: reasoning correctness and response naturalness are optimized as separate objectives, while adaptive reward weights coordinate the trade-off during training. The reasoning head is used as a training scaffold; evaluation reports reasoning accuracy, response quality, and joint success.

Run:

```bash
pip install -r requirements.txt
python run_pipeline.py
```

The production paper uses large dialogue models and online traffic; this toy version uses synthetic profile-conditioned e-commerce dialogues to preserve the optimization structure.
