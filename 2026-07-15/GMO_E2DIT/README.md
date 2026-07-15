# GMO-E2DIT Toy Reproduction

This folder implements a runnable PyTorch surrogate of **GMO-E2DIT: Grounded Multi-Operation Editing for E-Commerce Images**.

The paper's full system couples a VLM planner, a mask-conditioned image editor, and a reflection loop. This reproduction keeps the same pipeline interfaces while replacing foundation models with CPU-friendly neural modules:

- `data.py` creates toy e-commerce images with colored local patches and natural-language edit instructions.
- `model.py` implements an agenda planner, mask predictor, differentiable editor, and reflection head.
- `train.py` jointly trains operation classification, source/target mask grounding, edited-image reconstruction, and reflection success prediction.
- `test.py` runs an end-to-end edit session and reports operation accuracy, mask IoU, reconstruction error, and reflection accuracy.

Run:

```bash
pip install -r requirements.txt
python train.py --epochs 3
python test.py --checkpoint checkpoints/gmo_e2dit.pt
```

This is intentionally toy-scale; proprietary VLM/image-editor weights and EComEditBench data are replaced by synthetic image patches while preserving method logic.
