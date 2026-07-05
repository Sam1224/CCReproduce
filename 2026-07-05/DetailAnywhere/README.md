# DetailAnywhere Toy Reproduction

This folder reproduces the core pipeline of **DetailAnywhere: Fashion Detail Generation via Cross-Modal Feature Alignment Distillation**.

The original paper introduces Fashion Detail Generation, FDBench, Cross-modal Feature Alignment Distillation (CFAD), and a consistency reward model. This implementation keeps those interfaces in a runnable toy setting: a synthetic FDBench-style dataset supplies reference images, focus masks, and target detail crops; `DetailAnywhere` fuses reference and focus branches; `ViewBridgingTeacher` provides the distillation target; `detailanywhere_loss` combines reconstruction, CFAD, identity consistency, and reward objectives.

Run:

```bash
python train.py --cpu --epochs 1 --samples 32
python test.py --cpu --samples 16
```

The proprietary-scale diffusion backbone, DINOv3 teacher, FDBench images, and RLHF-style reward optimization are simplified because they require unpublished training data and large compute. The module boundaries are kept compatible with a full implementation.
