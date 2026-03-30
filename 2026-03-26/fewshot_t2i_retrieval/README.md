# Few-shot Text→Image Retrieval (toy reproduction)

Toy reproduction skeleton for:

- **Few Shots Text to Image Retrieval: New Benchmarking Dataset and Optimization Method ...** (arXiv:2603.25891)

## Idea captured

We implement a minimal dual-encoder retrieval pipeline:

- Contrastive pretraining on base classes.
- Few-shot adaptation on novel classes using a small number of paired examples.
- Evaluate retrieval **Recall@1** before/after adaptation.

## Quickstart

```bash
cd ccreproduce_repo/2026-03-26/fewshot_t2i_retrieval
python3 -m pip install -r requirements.txt
python3 train.py --pretrain-epochs 10 --adapt-epochs 5 --k 4
python3 test.py --k 4
```
