# RecoReward

PyTorch reproduction of the core RecoReward idea from `RecoReward: Recommender-Guided Multimodal Description Generation for Recommendation`.

The implementation keeps the paper's training/serving boundary: user-side behavior is only used by a frozen two-tower scorer during policy training, while the learned content policy performs content-only inference. The toy dataset simulates live-stream content vectors, target/non-target user embeddings, item descriptions, and a shared candidate bank.

## Files

- `dataset.py`: toy live-stream recommendation dataset aligned with the training and recall scripts.
- `model.py`: content policy, frozen scorer, Recommender Affinity Score, and group-relative policy loss.
- `train.py`: GRPO-style training loop using target/non-target RAS rewards.
- `test.py`: HR/NDCG/MRR recall evaluation against the toy candidate bank.

## Quick start

```bash
python train.py --epochs 3 --output recoreward_policy.pt
python test.py --checkpoint recoreward_policy.pt
```

The original paper uses MLLM generation, a production DSSM, Kuaishou behavior logs, and FSDP training. Those proprietary pieces are represented here by compatible modules and toy data interfaces so the pipeline can be run end-to-end locally.
