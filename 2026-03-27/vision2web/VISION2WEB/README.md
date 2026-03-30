# Vision2Web (toy reproduction)

This folder is a **runnable reproduction skeleton** for:

- **Vision2Web: A Hierarchical Benchmark for Visual Website Development with Agent Verification** (arXiv:2603.26648)

The paper frames website development as a *multimodal, long-horizon* agent problem: an agent must translate a visual/spec requirement into a working website, and verification should check correctness beyond “looks plausible”.

## What this reproduction covers

A tiny, fully-runnable benchmark loop that mirrors the paper’s *hierarchy + verification* idea:

- **Hierarchical spec**: a toy structured spec (layout flags + component counts).
- **Agent**: a simple PyTorch classifier that selects a generation “template” (4 templates) from spec features.
- **Website generator**: produces HTML/CSS from (spec, template).
- **Agent verification**: rule-based verifier that checks the generated HTML satisfies constraints (nav/search/footer/card count).

## What is NOT covered (paper-level TODOs)

- Real visual understanding (screenshots → code)
- Long-horizon editing with tool use (Playwright, DOM inspection, iterative fixes)
- Official dataset protocol and human evaluation

Those parts are left as TODO notes in the code.

## Quickstart

```bash
cd ccreproduce_repo/2026-03-27/vision2web/VISION2WEB
python3 -m pip install -r requirements.txt

# Build synthetic tasks
python3 build_toy_dataset.py --out /tmp/vision2web_train.json --n 500
python3 build_toy_dataset.py --out /tmp/vision2web_test.json --n 200 --seed 2

# Train template selector
python3 train.py --train /tmp/vision2web_train.json --out /tmp/vision2web.pt

# Evaluate generation + verification
python3 test.py --test /tmp/vision2web_test.json --ckpt /tmp/vision2web.pt
```

## Files

- `spec.py`: toy spec schema + feature encoding
- `build_toy_dataset.py`: synthetic task generator
- `model.py`: template selector (PyTorch)
- `generator.py`: HTML generation from spec
- `verifier.py`: constraint-based verification
- `train.py`: train selector
- `test.py`: evaluate pass@1 / constraint satisfaction
