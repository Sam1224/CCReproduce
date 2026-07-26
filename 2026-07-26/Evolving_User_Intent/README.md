# Evolving User Intent Toy Reproduction

Toy but runnable PyTorch reproduction for **LLMs Get Lost in Evolving User Intent**.

## What is reproduced

This folder keeps the paper's central evaluation story instead of rebuilding a full LLM agent stack:

- synthesize multi-turn user requests with three intent slots: `topic`, `tone`, and `budget`
- turn 0 gives a complete initial intent; later turns may revise one slot
- compare policies that are strong in static/no-drift settings with policies that explicitly track evolving intent
- report both ordinary static accuracy and evolving-intent accuracy, plus an intent-drift diagnostic

The expected qualitative result is the same as the paper's core warning: **a policy can look strong on static instruction following while failing after the user's intent changes over turns**.

## Files

- `dataset.py`: synthetic task data and user-intent evolution generator
- `agent.py`: static baseline, turn-only baseline, rule memory tracker, and tiny GRU neural memory policy
- `metrics.py`: static accuracy, evolving-intent accuracy, final-turn/drift-turn accuracy, slot accuracy, and drift scores
- `train.py`: trains the GRU memory policy on evolving dialogues and compares against rule baselines
- `test.py`: smoke/evaluation script showing the static-vs-evolving gap

## Run

```bash
python train.py --cpu --epochs 8 --samples 768
python test.py --cpu --checkpoint outputs/neural_memory.pt
```

Fast smoke test:

```bash
python train.py --cpu --epochs 1 --samples 64 --batch-size 16 --output-dir outputs_smoke
python test.py --cpu --samples 64 --checkpoint outputs_smoke/neural_memory.pt
```

## Expected behavior

A successful run should show:

- `static_initial` has perfect turn-0/static accuracy because it obeys the initial complete request
- the same `static_initial` policy drops on `evolving_acc` and especially final/drift turns when user intent changes
- `turn_only` uses the latest utterance but forgets older revisions, so it is better than static only in some cases
- `rule_memory_tracker` is near perfect because it explicitly updates slot memory
- `neural_memory` improves after training, demonstrating a learned lightweight memory policy

## Important differences from the original paper

- The original paper evaluates real LLM agents and natural language tasks; this reproduction uses synthetic slot intents.
- User turns are represented as compact PyTorch vectors instead of raw text prompts.
- The memory tracker is intentionally transparent so the failure mode is easy to inspect.
- Metrics are simplified but preserve the key diagnostic contrast: static accuracy is not enough for evolving user intent.
