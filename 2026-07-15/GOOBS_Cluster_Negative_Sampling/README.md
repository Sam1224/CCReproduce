# GOOBS / Cluster Negative Sampling Toy Reproduction

This folder implements a runnable PyTorch surrogate of **Real-Time Hard Negative Sampling via LLM-based Clustering for Large-Scale Two-Tower Retrieval**.

The reproduction mirrors the paper's core system:

- a two-tower user/item retrieval model;
- an online out-of-batch item pool;
- cluster-aware hard negative sampling where negatives come from the positive item's semantic cluster;
- a sampled-softmax style training objective;
- HR@K evaluation against all items in a toy catalog.

Run:

```bash
pip install -r requirements.txt
python train.py --epochs 4
python test.py --checkpoint checkpoints/goobs.pt
```

The LLM-based clustering in the paper is simulated by synthetic semantic cluster ids so the training/serving interface remains aligned but CPU-friendly.
