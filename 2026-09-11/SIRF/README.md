# SIRF: Spec-Internalized Risk Foundation Model

PyTorch reproduction scaffold for **SIRF: A Spec-Internalized Risk Foundation Model for Industrial Content Risk Control** (`arXiv:2609.11752v1`).

This implementation mirrors the paper's main pipeline with public/toy data interfaces:

- policy/spec tokens, content tokens, and account-level features;
- policy-grounded continued-pretraining losses for rule-node prediction and policy-path rationale supervision;
- verdict-only deployment head for White / Gray / Black content-risk decisions;
- train and test scripts that run end-to-end on synthetic data.

The original paper uses proprietary Xiaohongshu policy documents, account features, and disposition labels, so those pieces are represented by aligned toy interfaces here.

```bash
python train.py
python test.py
```
