# Self-Meta-Evolve toy reproduction

PyTorch implementation of the central mechanism from **One Prompt Does Not Fit All: Self-Meta-Evolve for Personalized Information Extraction**. The toy pipeline represents each enterprise persona, learns persona-specific extraction-field preferences, and keeps a trainable meta-prompt vector updated from successful prompt-edit memories.

Run:

```bash
python train.py
python test.py
```
