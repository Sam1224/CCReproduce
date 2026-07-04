# MIRAGE Toy Reproduction

This implementation mirrors MIRAGE with metadata-first grounding. The pipeline contains schema construction, metadata-guided image-text linking, metadata-conditioned answer generation, and a compact M-Eval style alignment report.

Run:

```bash
python train.py
python test.py
```

The paper uses enterprise KB chunks, curated images, Claude/Qwen-class metadata extraction, and LLM-as-judge evaluation. This reproduction replaces those proprietary components with deterministic toy schemas and a trainable metadata linker while preserving interfaces.
