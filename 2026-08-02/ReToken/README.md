# ReToken Reproduction

This folder contains a PyTorch reproduction of the core idea from **ReToken: One Token to Improve Vision-Language Models for Visual Retrieval**.

The original paper trains one learnable retrieval token plus a projection matrix on top of a mostly frozen VLM. The token retrieves query-relevant visual frames by cosine similarity against final-layer value vectors from a cached visual KV representation. This implementation keeps the same pipeline shape with a toy frozen VLM, a synthetic visual-haystack dataset, class-balanced retrieval loss, training script, and evaluation script.

## Files

- `model.py`: frozen toy VLM, value-cache retrieval, ReToken scoring, and class-balanced BCE loss.
- `dataset.py`: synthetic image/video haystack task with sparse relevant frames.
- `train.py`: trains the learnable ReToken and projection matrix.
- `test.py`: reports top-k retrieval recall and toy answer accuracy.

## Run

```bash
python train.py --epochs 5
python test.py --checkpoint checkpoints/retoken.pt --top-k 1
```

## Notes

The actual Qwen3VL/InternVL hidden states, KV-cache extraction, and full multimodal decoding are represented by `ToyFrozenVLM` so the repository remains runnable without large model weights. To align with the paper at production scale, replace `ToyFrozenVLM` with a VLM wrapper that exposes final-layer value vectors per frame and reuse `ReTokenRetriever.retrieval_loss` for supervised training.
