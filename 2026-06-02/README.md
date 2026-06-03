# Daily AI Paper Inspection — 2026-06-02

**Domain:** E-commerce content ecosystem & influencer (达人) governance  
**Emphasis:** LLM / VLM / MLLM, captioning, vector embeddings, clustering & similarity, agents, RAG, distillation, violation detection, content governance, data quality, data cleaning, large-scale data labeling

---

## Source Coverage

| source | category | attempted | http_status_or_error | candidates_yielded |
|--------|----------|-----------|----------------------|--------------------|
| arxiv cs.CL/new | aggregator | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| arxiv cs.CV/new | aggregator | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| arxiv cs.IR/new | aggregator | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| arxiv cs.MM/new | aggregator | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| arxiv cs.LG/new | aggregator | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| arxiv cs.AI/new | aggregator | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| HuggingFace papers/date/2026-06-02 | aggregator | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| HuggingFace papers/trending | aggregator | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| Google DeepMind blog | blog | yes | 403 Forbidden | 0 |
| Google Research blog | blog | yes | 403 Forbidden | 0 |
| Meta AI Research | blog | yes | 403 Forbidden | 0 |
| Anthropic News | blog | yes | 403 Forbidden | 0 |
| ByteDance Seed Research | blog | yes | 403 Forbidden | 0 |
| Meituan Tech | blog | yes | 403 Forbidden | 0 |
| Qwen Blog | blog | yes | 403 Forbidden | 0 |
| 量子位 QbitAI | wechat | yes | 403 Forbidden | 0 (fallback: WebSearch found no June-2 articles) |
| 机器之心 (WebSearch fallback) | wechat | yes | no direct result for 2026-06-02 | 0 |
| 新智元 (WebSearch fallback) | wechat | yes | no direct result for 2026-06-02 | 0 |
| OpenAI News (WebSearch fallback) | blog | yes | WebSearch OK | 0 (no research papers on 2026-06-02, only product announcements) |
| DeepSeek (WebSearch fallback) | blog | yes | no 2606 papers found | 0 |
| Xiaohongshu / RedTech (WebSearch fallback) | blog | yes | no result | 0 |
| Kuaishou Tech (WebSearch fallback) | blog | yes | no result | 0 |
| Tencent Hunyuan (WebSearch fallback) | blog | yes | no result | 0 |
| Semantic Scholar Graph API | aggregator | yes | 403 Forbidden | 0 |
| arxiv WebSearch (2606 prefix — cs.CL/CV/IR) | aggregator | yes | WebSearch OK | 5 candidates |
| OpenReview ICLR/NeurIPS 2026 | conf | no | not yet active | 0 |

> **Note on arXiv access:** All direct `/list/` and `/html/` and `/abs/` WebFetch calls returned HTTP 403 for this session. Discovery relied entirely on WebSearch using `"arxiv.org/abs/2606"` patterns and targeted keyword searches; all candidate papers were verified via multiple independent WebSearch queries before inclusion.

---

## Paper Summary — Picked Papers

| # | Title | arXiv ID | Bucket | Score |
|---|-------|----------|--------|-------|
| 1 | PaSBench-Video: A Streaming Video Benchmark for Proactive Safety Warning | 2606.02443 | STRONG | **73** |
| 2 | ExpWeaver: LLM Agents Learn from Experience via Latent RAG | 2606.01041 | WEAK | **72** |
| 3 | RCEM: Embedder Equipped with Query Rewriting Skill for Robust Conversational Search | 2606.01697 | WEAK | **66** |
| 4 | CRAM: Centroid-Routing and Adaptive MoE for Multimodal Continual Instruction Tuning | 2606.02502 | WEAK | **65** |
| 5 | Moment-Video: Diagnosing Temporal Fidelity of Video MLLMs on Momentary Visual Events | 2606.02522 | STRONG | **64** |

> No paper scored ≥ 80; code reproduction folder was not created for this day.

---

## 1. PaSBench-Video (Score: 73) — STRONG

**Story arc:** Existing video benchmarks rely on static inputs, ignore timing precision, and lack false-positive measurement — meaning no benchmark tests whether MLLMs can proactively warn about danger *before* harm occurs. PaSBench-Video fills this gap: 740 streaming-video clips, 13 MLLM evaluations, progressively stricter metrics (hit → timed-hit → content-correct). Result: no model exceeds 20% on the strictest metric, Pearson r = 0.64 between recall and false-positive rate.

**Key metrics:**
- `PaSBench-Video (strictest)` `HitRate@content` best = `Seed-2.0-Pro: <20%`
- `PaSBench-Video` `Recall/FP Pearson r` `0.64` — models trade recall for false positives, no free lunch

**Link:** https://arxiv.org/abs/2606.02443 | Paper file: `papers/pasbench_video.md`

---

## 2. ExpWeaver (Score: 72) — WEAK

**Story arc:** Text-based RAG for agents requires large context windows and separates retrieval from generation, creating token overhead and architectural mismatch. ExpWeaver replaces explicit text retrieval with latent-space retrieval: experience hidden states retrieved by cross-attention at *every decoding step*, gated residual integration, end-to-end RL training. No separate RAG module. SOTA on 12/13 tasks, >6.8% over baselines.

**Key metrics:**
- `13-task benchmark (QA, reasoning, coding, recommendation)` `Avg accuracy` `SOTA on 12/13` vs baselines `>6.8% below`

**Link:** https://arxiv.org/abs/2606.01041 | Paper file: `papers/expweaver.md`

---

## 3. RCEM (Score: 66) — WEAK

**Story arc:** Conversational retrieval models suffer under distributional shift (train on one conversation style, deploy on another). Prior work requires conversation-to-document relevance pairs, which are expensive. RCEM distills LLM query-rewriting capability into the embedding model via alignment training — at inference, no rewriting needed; the embedding itself carries context awareness. Up to 20% Recall@10 gain under shift.

**Key metrics:**
- `QReCC` `Recall@10` improvement `up to 20%` vs strong baselines
- `TopiOCQA, TREC CAsT` consistent outperformance

**Link:** https://arxiv.org/abs/2606.01697 | Paper file: `papers/rcem.md`

---

## 4. CRAM (Score: 65) — WEAK

**Story arc:** Multimodal continual instruction tuning faces a dilemma: shared parameters → catastrophic forgetting; dedicated-module-per-task → parameter explosion over long streams. CRAM resolves this via centroid-routing (tasks cluster to experts) + adaptive-rank MoE instantiation (only allocates parameters proportional to the capability gap). Beats both extremes.

**Key metrics:**
- Multiple MCIT tasks: outperforms shared-parameter and dedicated-module baselines on all
- Adaptive rank allocation: reduces parameters vs fixed-rank MoE

**Link:** https://arxiv.org/abs/2606.02502 | Paper file: `papers/cram.md`

---

## 5. Moment-Video (Score: 64) — STRONG

**Story arc:** Video MLLMs score well on global scene understanding but their ability to preserve brief, answer-critical evidence is untested. Moment-Video: 1,000 QA pairs, 7 domains, momentary events (few-frame actions/state transitions). Best model (Seed-2.0-Pro) scores only 39.6%; open-source models mostly <25%.

**Key metrics:**
- `Moment-Video` `Overall Acc` best proprietary `39.6% (Seed-2.0-Pro)` vs most open-source `<25%`
- `GPT-5.4` `Overall Acc` `26.9%`; `Gemini-series` `20.4–26%`

**Link:** https://arxiv.org/abs/2606.02522 | Paper file: `papers/moment_video.md`
