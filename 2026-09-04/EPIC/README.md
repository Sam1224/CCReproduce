# EPIC toy reproduction

This folder provides a **toy but runnable** PyTorch reproduction of **EPIC: Explicit Posterior Item Conditioning for Semantic ID Diffusion Recommendation**.

## What is implemented
- Synthetic sequential recommendation dataset with catalog items represented by 4-position semantic IDs
- Frozen masked-SID backbone that predicts SID tokens and item logits from user histories
- EPIC-style adapter that builds a feasible candidate set from partial SIDs
- Personalized item posterior from recent-history transition evidence
- Projection of the item posterior back to SID positions during denoising
- End-to-end evaluation with Recall / NDCG against a baseline decoder

## What is simplified
- The masked diffusion process is approximated by iterative greedy denoising over four SID positions.
- Transition evidence is computed from learned recent-history embeddings rather than the full paper's candidate-conditioned residual design.
- Amazon public benchmarks are replaced by a synthetic catalog whose generation process preserves the paper's key reasoning structure.

## Files
- `data.py`: synthetic catalog, semantic IDs, and user-sequence generation
- `model.py`: frozen SID backbone, explicit posterior item conditioning adapter, ranking metrics
- `train.py`: pretrains the backbone and then trains only the EPIC adapter
- `test.py`: compares baseline decoding with EPIC-enhanced decoding

## Run
```bash
python train.py
python test.py
```

## Expected outcome
The EPIC-enhanced decoder should outperform the baseline on synthetic `Recall@5` / `NDCG@5`, illustrating the paper's core claim: **injecting personalized item posteriors during denoising preserves promising item hypotheses better than token-only decoding**.
