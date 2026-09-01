# ConfigurableSemanticChunking

Toy but end-to-end PyTorch reproduction for **Configurable Semantic Chunking for Biomedical Information Extraction in Retrieval-Augmented Generation** (arXiv:2608.31139).

The paper replaces fixed-size RAG chunks with configurable, entity-preserving, trigger-centered, proposition-first chunks and uses relation hierarchy rules to rank evidence. This implementation keeps the same abstraction but uses a small synthetic e-commerce/governance corpus so the pipeline can run quickly in this repository.

## What is reproduced

- Entity-preserving candidate windows.
- Trigger-centered chunk creation with configurable trigger tiers.
- Proposition-first subject-trigger-object extraction.
- Relation hierarchy scoring and tie-breaking.
- A lightweight PyTorch chunk scorer trained on toy query/chunk relevance pairs.
- End-to-end retrieval evaluation against fixed-size chunking.

## Files

- `data.py`: toy documents, query labels, and corpus helpers.
- `model.py`: configuration objects, semantic chunker, resolver, and PyTorch scorer.
- `train.py`: scorer training loop and feature generation.
- `test.py`: retrieval metrics and comparison runner.
- `run_pipeline.py`: one-command end-to-end demo.
- `requirements.txt`: runtime dependency list.

## Quickstart

```bash
pip install -r requirements.txt
python run_pipeline.py
python test.py
```

Expected output shows the learned semantic chunker returning higher top-1 retrieval accuracy than the fixed-size baseline on the toy corpus.

## Notes

The original paper evaluates biomedical GM-CIHT, DDI, ChemProt, and ADE datasets, reporting 82.6% F1 on GM-CIHT versus 74.2% for a fixed-size baseline. This reproduction mirrors the method design, not the full biomedical datasets or proprietary BioMedRAG scorer. Replace `build_toy_corpus()` with real domain documents and tune `SemanticChunkConfig` for production use.
