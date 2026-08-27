# CEMMKG toy reproduction

This folder implements a runnable PyTorch reproduction of **Multi-Granularity Context-Enhanced RAG over Multimodal Knowledge Graphs**.

The toy pipeline keeps the paper's core logic:

1. build visual elements from synthetic ecommerce-like documents;
2. create local context from surrounding and semantically related text;
3. create global document context;
4. inject context into image-to-graph construction and cross-modal fusion;
5. train a context-aware classifier as a proxy for MMKG-RAG answer selection.

Run:

```bash
python train.py
python test.py
```

Large MLLM graph extraction, PDF parsing, and production MMKG storage are represented by deterministic tensor interfaces so the method can run on CPU in CI.
