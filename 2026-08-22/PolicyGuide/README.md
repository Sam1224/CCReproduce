# PolicyGuide reproduction

This folder implements a compact PyTorch-compatible reproduction of the core PolicyGuide idea: compile a policy into an explicit workflow graph, maintain completed workflow state, and proactively block or guide the next action when prerequisites are missing.

Files:

- `dataset.py`: toy retail and creator-governance policy cases.
- `model.py`: workflow graph, proactive verifier module, and remediation logic.
- `train.py`: trains a small neural verifier to predict next required policy step from completed-step state.
- `test.py`: validates graph-level blocking and completed-workflow allowance.

Run:

```bash
python train.py
python test.py
```

The reproduction keeps the key paper abstraction while replacing unavailable proprietary τ2-bench environments and closed-model verifier calls with transparent toy cases and a small neural classifier.
