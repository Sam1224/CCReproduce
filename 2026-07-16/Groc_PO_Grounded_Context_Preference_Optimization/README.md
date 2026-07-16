# Groc-PO Toy Reproduction

This folder implements a runnable PyTorch surrogate of **Groc-PO: Grounded Context Preference Optimization for Truthful Multimodal LLMs**.

The paper proposes stage-aware preference optimization for MLLMs: instead of only preferring the final answer, it supervises object grounding, contextual grounding, and grounded reasoning so early grounding drift does not propagate into hallucinated answers. This toy implementation keeps the method contract while replacing the large MLLM with CPU-friendly encoders:

- `data.py` creates synthetic multimodal preference pairs with image attributes, text context, stage labels, chosen responses, and rejected responses.
- `model.py` implements a tiny grounded-context reward model plus a Groc-PO/DPO loss applied at each grounding stage.
- `train.py` optimizes stage-wise preference margins and reports held-out chosen-vs-rejected accuracy.
- `test.py` runs an end-to-end toy audit session and prints stage-level preference accuracy and hallucination-risk rate.

Run:

```bash
pip install -r requirements.txt
python train.py --epochs 3
python test.py --checkpoint checkpoints/groc_po.pt
```

This is intentionally toy-scale; proprietary MLLM backbones, large preference data, and benchmark infrastructure are replaced by synthetic tensors while preserving the staged preference-optimization pipeline.
