# DART-SD toy reproduction

PyTorch reproduction of the core algorithmic ideas in `DART-SD: Diamond-topology Aware Retrieval and Tuning for Self-Distillation of Multi-Turn Tool-Calling Agents`.

Implemented pieces:
- Interaction-State Transition Graph over cumulative information atoms.
- Success-reachable projection and Critical Topological Breakpoint detection.
- Recovery suffix retrieval from successful teacher traces.
- CTB-localized language-model loss that masks valid prefixes and optimizes only recovery tokens.
- Toy dataset, training script, and smoke test.

Run:

```bash
python test.py
python train.py
```
