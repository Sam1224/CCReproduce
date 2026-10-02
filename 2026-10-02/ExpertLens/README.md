# ExpertLens reproduction

PyTorch reproduction of **Harnessing Domain Specialists in Multimodal Mixture-of-Experts for Efficient Adaptation**.

The linked GitHub page was unavailable/404 at inspection time, so this folder implements the core ideas:

- a small multimodal MoE block with token routing;
- data-free `ExpertLens` decoding of router/expert weights into semantic vocabulary tokens;
- domain expert selection from decoded semantics;
- selective fine-tuning of only the selected experts on toy multimodal classification tasks.

## Files

- `expertlens.py`: MoE model, semantic lens, selective-freeze helpers.
- `toy_data.py`: toy visual/text domain features for e-commerce, medical, math, remote sensing.
- `train.py`: trains base MoE and adapts selected experts.
- `test.py`: validates expert selection and selective adaptation.

## Quick start

```bash
pip install torch
python train.py
python test.py
```
