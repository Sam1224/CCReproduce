# Customer-Agent — Toy Reproduction

Paper: Customer-Agent: Overcoming Context Limitations in Ultra-Long Shopping Trajectories via Tool-Augmented Agents and RLVR  
arXiv: https://arxiv.org/abs/2606.07995

This reproduction keeps the paper's key structure: shopping trajectories are stored as external files, a small agent learns tool/SQL-template selection from questions, and evaluation compares direct answer prediction with tool-verified execution.

Run:

```bash
pip install -r requirements.txt
python run_pipeline.py
```

The real paper trains large LLM agents with SFT + RLVR and code-interpreter feedback; this toy version uses synthetic CSV trajectories and verifiable SQL-style templates to make the pipeline runnable on CPU.
