# WebRetriever Toy Reproduction

This folder reproduces the core pipeline of **WebRetriever: A Large-Scale Real-World Benchmark for Web Agent Navigation and Task Completion** in a runnable toy setting.

The public release describes a benchmark that separates correct-page arrival from actual task completion on live websites. This implementation keeps that interface: a synthetic web-task dataset provides page graphs, target pages, evidence nodes, and action traces; `WebRetrieverAgent` encodes query and page states, retrieves candidate pages, predicts an action sequence, and estimates final completion. The loss combines page retrieval, action imitation, and completion calibration.

Run:

```bash
python train.py --cpu --epochs 1 --samples 64
python test.py --cpu --samples 32
```

The real benchmark's live websites, browser execution harness, and private baseline outputs are simplified into deterministic toy page graphs. The module boundaries are designed so a full implementation can replace the synthetic dataset with browser traces and HTML embeddings.
