# CMDR-Embed Toy Reproduction

This folder implements a compact PyTorch reproduction of CMDR-Embed: pages are encoded jointly with document context, split back into page-level embeddings, and trained with a retrieval objective plus a same-document hard-negative CMCL-style penalty.

Run:

```bash
python train.py --cpu
python test.py --cpu
```

The toy dataset mimics CMDR-Bench by making each query depend on both a relevant page and a neighboring context page.
