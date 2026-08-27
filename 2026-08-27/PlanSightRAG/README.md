# PlanSightRAG toy reproduction

This folder implements a runnable PyTorch reproduction of **PlanSightRAG: A Visual-First Multimodal RAG for Automating Question Answering and Compliance Checking for Civil Standard Plans**.

The toy version keeps the core pipeline:

1. synthetic plan pages are represented as visual patch embeddings;
2. a query is encoded and matched to page patches with MaxSim late interaction;
3. top-k retrieval provides a heatmap-like evidence tensor;
4. a lightweight compliance auditor compares retrieved rule values with proposed drawing values.

Run:

```bash
python train.py
python test.py
```

The real paper uses ColNomic-3B, Qwen2.5-VL, high-resolution plans, and an agentic Planner-Retriever-Auditor-Synthesizer loop. Those heavy components are replaced by CPU-friendly interfaces here while retaining the data/model/train/test contract.
