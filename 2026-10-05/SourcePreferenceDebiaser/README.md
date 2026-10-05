# SourcePreferenceDebiaser

Toy but runnable PyTorch reproduction for **Source Preference in the Wild: How LLM Agents Favor Items by Source, and How to Reduce It**.

The paper studies source bias in agentic search: agents select items from preferred sources even when the item satisfies fewer requirements. This reproduction implements the core experimental logic:

- a synthetic shopping/accommodation/scholar-search style dataset;
- pairwise ranking over items with requirement-match features and source IDs;
- source preference measurement under matched utility and position controls;
- mitigation through adversarial source removal plus counterfactual relabel consistency.

## Files

- `dataset.py` builds toy matched-pair and counterfactual examples.
- `model.py` defines the ranker, gradient reversal layer, and debiasing loss.
- `train.py` trains the biased baseline and debiased model.
- `test.py` reports utility accuracy and source-preference gaps.

## Quick start

```bash
python train.py --epochs 12 --output-dir runs/demo
python test.py --checkpoint runs/demo/debiased.pt
```
