# E-GEO Toy Reproduction

A compact PyTorch reproduction of E-GEO's e-commerce generative engine optimization loop.

The toy pipeline builds query/listing pairs, trains a lightweight LLM-reranker analogue, and learns a prompt-policy mixture that rewrites product descriptions to improve rank while penalizing manipulative language.

## Run

```bash
pip install -r requirements.txt
python train.py --epochs 3
python test.py --checkpoint checkpoints/egeo_ranker.pt
```
