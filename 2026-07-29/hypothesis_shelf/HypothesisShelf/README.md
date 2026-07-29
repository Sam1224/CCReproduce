# HypothesisShelf

PyTorch reproduction of the core pipeline from `Hypothesis-Driven Shelf Generation for Personalised Recommendation`.

The implementation follows the paper's four-stage decomposition: user-profile hypothesis planning, constrained catalogue fulfilment, candidate selection and shelf alignment, and offline serving-compatible output records. It uses toy catalogue/profile data but keeps interfaces aligned across data, model, training, and evaluation scripts.

## Files

- `dataset.py`: synthetic user profiles, typed catalogue entities, and positive shelf items.
- `model.py`: hypothesis planner, constrained generative-style fulfilment over typed semantic IDs, and shelf aligner.
- `train.py`: end-to-end training for type prediction, fulfilment, and alignment.
- `test.py`: type accuracy and final shelf hit-rate evaluation.

## Quick start

```bash
python train.py --epochs 3 --output hypothesis_shelf.pt
python test.py --checkpoint hypothesis_shelf.pt
```

The production paper uses Spotify-scale profiles, catalogue entities, LLM-generated shelf text, SemID tries, and online randomised exposure. This reproduction provides runnable modules with matching boundaries so the method can be adapted to a real catalogue later.
