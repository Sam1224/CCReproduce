# VDA toy reproduction

This folder implements a runnable PyTorch reproduction of **A Visual Dependence-Aware Framework for Multimodal Unsupervised Continual Post-Training**.

The toy pipeline preserves the key ideas:

1. sequential multimodal tasks arrive without answer labels for the language-modeling objective;
2. token-level visual dependence is estimated from token-region attention;
3. visually modulated adaptation upweights visually grounded tokens;
4. a simplified VC-OT regularizer preserves old-task visual-dependence strength and structure using replay batches.

Run:

```bash
python train.py
python test.py
```

The full paper uses Qwen2.5-VL-7B with LoRA and six multimodal benchmarks. This reproduction uses synthetic image-region and token tensors to keep the algorithmic interfaces runnable on CPU.
