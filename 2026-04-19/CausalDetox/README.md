# CausalDetox (Toy Reproduction)

Paper: **CausalDetox: Causal Head Selection and Intervention for Language Model Detoxification** (arXiv:2604.14602)

This is a *toy* implementation that captures the paper’s core pipeline shape:

1) train a small Transformer LM on synthetic text where a dedicated token represents a toxic continuation
2) perform **component (head) selection** by estimating a necessity-style effect: how much ablating a head reduces the toxic-token probability
3) apply an **inference-time intervention** by scaling down selected heads

It is not a faithful reproduction of PNS estimation, confounder modeling, or PARATOX; those parts would require a larger model and careful causal setup. The goal is to provide an executable scaffold for “head selection → intervention → detox effect”.

## Run

```bash
pip install torch
python train.py
python test.py
```
