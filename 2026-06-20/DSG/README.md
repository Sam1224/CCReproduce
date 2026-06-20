# DSG — Decoupled Search Grounding (toy, offline reproduction)

Self-contained reproduction of the **systems architecture** in:

> Emmanuel Aboah Boateng, Kyle MacDonald, Amardeep Kumar, Siddharth Kodwani, Sudeep Das.
> *Decoupling Search from Reasoning: A Vendor-Agnostic Grounding Architecture for LLM Agents* (DoorDash).
> arXiv: https://arxiv.org/abs/2606.18947

This is a **systems / architecture** paper (no neural net). The paper ships no code,
so this folder implements the DSG gateway faithfully at toy scale, fully offline
(no network, no model downloads). Everything runs in seconds on CPU.

## What is implemented (1:1 with the paper)

- **Provider Registry + configured fallback chain** (Sec 4.2) — `ProviderRegistry`
  with an ordered chain `Serper → BrightData → NativeSearch`. A provider
  timeout/error advances the request along the chain (`fail_rate` simulates
  transient failures deterministically).
- **Cache-gated execution — Algorithm 1** (`DSGGateway.query`):
  `Normalize → ExactCache → SemanticCache (cos ≥ τ) → provider fallback`,
  then write-back to both caches and `RenderContext`.
- **Semantic cache via cosine of query embeddings** (Eq. 1) — `embeddings.embed`
  is an offline word-level hashing bag-of-words (the "simple embedding fallback");
  `cos(e(q), e(q_j)) ≥ τ` gates reuse.
- **Provider-scoped cache keys + per-domain TTL** (Sec 4.3) — keys are
  `provider::normalized_query`; `DOMAIN_TTL` expires recency-sensitive domains
  (weather/finance) fast while static factoids persist.
- **Source-aware context rendering** (Sec 4.1) — `render_context` pairs every
  snippet with its source URL for explicit provenance.
- **Telemetry** (Sec 4.3) — selected provider, retrieval depth, cache outcome,
  latency, cost, semantic score, and the fallback path per request.
- **Retrieval-depth knob** (Fig 4A) — `max_results` on the gateway.

## Data (`data.py`)

- A tiny grounding corpus (`Document`: url, domain, snippet, answer, keywords).
- A factoid query set (SimpleQA/FreshQA/HotpotQA flavour) **and** an e-commerce
  **QIU**-style query-intent set, plus paraphrases used to drive semantic-cache reuse.

## Train / calibrate (`train.py`)

Calibrates the semantic-cache threshold **τ** on the toy set: paraphrase pairs are
positives (should reuse cache), distinct queries are negatives. Sweeps τ to maximise
balanced accuracy and writes `tau.json`.

## Test / eval (`test.py`)

Runs a **cold** pass (fills caches via the fallback chain) and a **warm** replay,
then reports accuracy, cache-hit rate, latency, and cost vs a **native-search**
baseline — mirroring the paper's headline metrics (warm-cache hit, latency
reduction, cost reduction).

## Run

```bash
pip install -r requirements.txt
python train.py     # calibrate tau -> tau.json
python test.py      # cold/warm eval + telemetry
```

Example output: τ=0.50; DSG accuracy 100% (== native), warm-cache hit 100%,
latency reduction ~100% on cache hits, search-cost reduction ~96% vs native.

## Approximations (and why)

| Paper | Here |
|---|---|
| Frontier LLM reasoners (GPT-4o, Gemini, Claude) | a trivial "top-result answer" extractor — the gateway, not the reasoner, is the contribution |
| Real providers (Serper/BrightData/Firecrawl/Exa) | `CorpusProvider` over a local corpus, with deterministic transient-failure simulation to exercise fallback |
| Real embedding model | offline word-hashing bag-of-words (`embeddings.embed`) |
| Real provider $ / latency | toy constants ordered like the paper (cheap-but-flaky first, expensive-reliable last) |

The cache logic, fallback policy, provider abstraction, per-domain TTL,
source-aware rendering, telemetry, and τ-calibration are faithful and drop-in
replaceable with real providers / embedders.

## Files
- `data.py` — toy corpus + factoid/QIU queries + paraphrases
- `embeddings.py` — offline hashing embedding + cosine
- `gateway.py` — providers, registry+fallback, exact/semantic caches (Algorithm 1), TTL, rendering, telemetry
- `train.py` — semantic-cache τ calibration
- `test.py` — cold/warm evaluation vs native baseline
