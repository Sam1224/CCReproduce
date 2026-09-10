# Glyph: Multi-Strategy Agentic Data Catalog Tagging

This folder contains a compact PyTorch reproduction of the core ideas from **Glyph: A Multi-Strategy Agentic System for Column Description and Sensitivity-Ontology Tagging of Enterprise Data Catalogs**.

The implementation mirrors the paper's production design at toy scale:

- a metadata encoder trained with a contrastive objective for same-tag retrieval;
- a multi-label neural classifier for ontology tags;
- rule-based, description-based, and metadata-neural strategies;
- Reciprocal Rank Fusion with per-strategy provenance;
- a runnable synthetic data pipeline aligned with enterprise catalog keys.

## Files

- `data.py`: ontology, synthetic catalog examples, tokenizer, and dataset.
- `model.py`: metadata encoder, tagger, regex/description taggers, and fusion pipeline.
- `train.py`: training loop with multi-label and contrastive losses.
- `test.py`: smoke test for end-to-end inference.
- `requirements.txt`: minimal dependencies.

## Run

```bash
pip install -r requirements.txt
python train.py --epochs 2 --cpu
python test.py
```

The original paper relies on private production catalog metadata, source-code retrieval, and steward labels. This reproduction therefore uses synthetic toy catalog data while preserving the interfaces and learning/fusion mechanics needed to swap in real catalog records.
