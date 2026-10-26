# EmbeddingModelV3: Reproduction for Structured LLM-Annotated Dense Retrieval

This is a lightweight PyTorch-oriented reproduction of:

**Scaling Dense Retrieval with LLM-Annotated Training Data: Structured Mining and Progressive Curriculum for E-Commerce Sponsored Search**

The original paper focuses on production-scale sponsored-search retrieval at Walmart. This reproduction keeps the paper's core structure while replacing heavy private components with deterministic, swappable modules:

- multi-channel retrieval disagreement mining;
- calibrated annotation cascade (`184M cross-encoder -> 2B LLM -> 8B LLM`) represented by a pluggable toy cascade;
- five sample difficulty buckets;
- three-stage curriculum (`BCE -> MNR -> Triplet`);
- Siamese two-tower dense retriever.

## Files

- `data.py` — toy sponsored-search corpus and curriculum mining utilities.
- `model.py` — tokenizer, two-tower retriever, loss functions, toy cascade annotator.
- `train.py` — end-to-end reproduction script.
- `test.py` — smoke evaluation script for checkpoint loading and ranking quality.
- `requirements.txt` — minimal dependency list.

## Run

```bash
pip install -r requirements.txt
python train.py
python test.py
```

## What is faithfully reproduced

- The paper's **upstream-data-first** logic.
- Sample construction from **multi-channel agreement/disagreement**.
- A **cascade annotation** abstraction with confidence routing.
- A **progressive curriculum** from easy classification to hard contrastive separation.
- A **shared-encoder two-tower retriever** using cosine similarity.

## What is approximated

- The real 184M/2B/8B annotators are replaced by `ToyCascadeAnnotator`.
- The real production channels (dictionary/BM25/ANN) are represented by toy channel/rank metadata.
- The paper's 240M+ examples / 4M queries are replaced by a compact toy dataset.

## Extension points

- Replace `ToyCascadeAnnotator.annotate()` with real cross-encoder / LLM annotators.
- Replace `build_raw_records()` in `data.py` with mined production retrieval logs.
- Replace the tokenizer with a SentenceTransformer or BERT tokenizer while keeping the same trainer.
