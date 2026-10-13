# ECC toy reproduction

This folder provides a **toy but runnable** PyTorch reproduction of **ECC: Evolving Creatives with Clicks**.

## What is implemented
- Synthetic e-commerce ad creative dataset with editable visual attributes
- CTR-sensitive attribute mining with a lightweight ANOVA-style effect score
- Instruction-tuning dataset construction from low/high CTR creative pairs
- A PyTorch seq2seq suggestion model that generates executable edit operations
- CPO-style preference optimization with:
  - chosen vs rejected suggestion preference loss
  - consistency regularization between chosen suggestions and shuffled-equivalent variants
- End-to-end scripts for training and evaluation

## What is simplified
- Real image editing is replaced by structured attribute editing on toy creatives.
- MLLM-based attribute extraction is approximated by directly observed toy attributes.
- Real online CTR traffic feedback is simulated by a synthetic reward function over attributes.
- The paper's large-scale preference collection is replaced by controllable toy preference triples.

## Files
- `data.py`: synthetic creative generation, attribute mining, SFT/preference data builders
- `model.py`: tokenizer, seq2seq suggestion model, decoding, preference utilities
- `train.py`: SFT training + CPO preference optimization
- `test.py`: checkpoint loading and held-out evaluation
- `run_pipeline.py`: one-command training + testing
- `requirements.txt`: minimal runtime requirements

## Run
```bash
python train.py
python test.py
```

or

```bash
python run_pipeline.py
```

## Expected outcome
On the synthetic benchmark, the suggestion model should learn to:
1. identify CTR-harmful attributes,
2. generate structured edit suggestions,
3. improve preference-aligned decoding after the CPO phase.

## Notes on faithfulness
This reproduction preserves the paper's **algorithmic skeleton**:
- mine CTR-sensitive attributes,
- learn edit suggestions from high/low CTR contrasts,
- align suggestions with click preferences,
- regularize semantically equivalent edit-order variants.

The unavailable parts are explicitly simplified rather than silently omitted.
