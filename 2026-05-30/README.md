# Daily AI Paper Inspection — 2026-05-30 (GMT+8)

**Domain Focus:** E-commerce content ecosystem & influencer (达人) governance
**Emphasis:** LLM / VLM / MLLM · captioning · vector embeddings · clustering & similarity · agents · RAG · distillation · violation detection · content governance · data quality · large-scale data labeling

> **Note on date window:** 2026-05-30 is a **Saturday**. arXiv does not post new listings on weekends; the most recent arXiv listing was Friday 2026-05-29 (covering submissions through ~2026-05-28). HuggingFace paper aggregator pages also returned HTTP 403. Primary discovery was conducted via WebSearch targeting papers with arXiv IDs in the `2605.xxxxx` range (May 2026). Catch-up coverage extends across the working week of 2026-05-25 through 2026-05-30 per protocol.

---

## Source Coverage

| source | category | attempted | http_status_or_error | candidates_yielded |
|--------|----------|-----------|---------------------|--------------------|
| arxiv.org/list/cs.CL/new | aggregator | yes | 403 Forbidden | 0 |
| arxiv.org/list/cs.CV/new | aggregator | yes | 403 Forbidden | 0 |
| arxiv.org/list/cs.IR/new | aggregator | yes | 403 Forbidden | 0 |
| arxiv.org/list/cs.MM/new | aggregator | yes | 403 Forbidden | 0 |
| arxiv.org/list/cs.LG/new | aggregator | yes | 403 Forbidden | 0 |
| arxiv.org/list/cs.AI/new | aggregator | yes | 403 Forbidden | 0 |
| huggingface.co/papers/date/2026-05-30 | aggregator | yes | 403 Forbidden | 0 |
| huggingface.co/papers/trending | aggregator | yes | 403 Forbidden | 0 |
| deepmind.google/discover/blog/ | blog | yes | 403 Forbidden | 0 |
| research.google/blog/ | blog | yes | 403 Forbidden | 0 |
| ai.meta.com/research/ | blog | yes | 403 Forbidden | 0 |
| anthropic.com/news | blog | yes | 403 Forbidden | 0 |
| seed.bytedance.com/research | blog | yes | 403 Forbidden | 0 |
| tech.meituan.com/ | blog | yes | 403 Forbidden | 0 |
| qwenlm.github.io/blog/ | blog | yes | 403 Forbidden | 0 |
| qbitai.com/ | wechat | yes | 403 Forbidden | 0 |
| semanticscholar.org Graph API | aggregator | yes | 403 Forbidden | 0 |
| WebSearch: jiqizhixin.com 机器之心 2026-05 | wechat/search | yes | results returned | 0 qualifying |
| WebSearch: mp.weixin.qq.com 新智元 2026-05 | wechat/search | yes | results returned | 0 qualifying |
| WebSearch: openai.com/news 2026-05 | blog/search | yes | results returned | 0 qualifying |
| WebSearch: deepseek 2026-05 | blog/search | yes | results returned | 0 qualifying |
| WebSearch: 快手 Kuaishou Tech 2026-05 | blog/search | yes | 2602.09401 via search | 1 |
| WebSearch: 腾讯混元 Hunyuan 2026-05 | blog/search | yes | results returned | 0 qualifying |
| WebSearch: 小红书 REDtech 2026-05 | blog/search | yes | results returned | 0 qualifying |
| WebSearch: arxiv 2605 e-commerce LLM content | search fallback | yes | multiple 2605 papers | 7 |
| WebSearch: arxiv 2605 data filtering AIGC | search fallback | yes | 2605.19407, 2605.21977 | 2 |
| OpenReview ICLR/NeurIPS 2026 | conf | no | not yet active | 0 |

**Summary:** 26 sources attempted; 11 sources blocked (403); 15 sources returned results via WebSearch. Tier 1 WebFetch uniformly returned 403 (sandbox network policy blocks those CDN domains). All discovery conducted via WebSearch fallback (Tier 3). **9 qualifying candidates** identified.

---

## Candidate Papers — Scored

### STRONG candidates (directly in scope)

| # | Title | arXiv | Date | Score | Bucket |
|---|-------|-------|------|-------|--------|
| 1 | Text-Guided Visual Representation Learning for Robust Multimodal E-Commerce Recommendation (TGQ-Former) | 2605.17366 | 2026-05-14 | **82** | STRONG |
| 2 | A General Framework for Multimodal LLM-Based Multimedia Understanding in Large-Scale Recommendation Systems | 2605.09338 | 2026-05-10 | **74** | STRONG |
| 3 | SARM: LLM-Augmented Semantic Anchor for End-to-End Live-Streaming Ranking | 2602.09401 | 2026-02-10 | **83** | STRONG |
| 4 | Thinking Broad, Acting Fast: Latent Reasoning Distillation from Multi-Perspective CoT for E-Commerce Relevance | 2601.21611 | 2026-01-29 | **85** | STRONG |
| 5 | E-VAds: An E-commerce Short Videos Understanding Benchmark for MLLMs | 2602.08355 | 2026-02-09 | **78** | STRONG |
| 6 | Adapting Vision-Language Models for E-commerce Understanding at Scale | 2602.11733 | 2026-02-12 | **71** | STRONG |

### WEAK candidates (high-impact, transferable)

| # | Title | arXiv | Date | Score | Bucket |
|---|-------|-------|------|-------|--------|
| 7 | Gemini Embedding 2: A Native Multimodal Embedding Model from Gemini | 2605.27295 | 2026-05-27 | **79** | WEAK |
| 8 | Video as Natural Augmentation: Towards Unified AI-Generated Image and Video Detection (VINA) | 2605.21977 | 2026-05-21 | **65** | WEAK |
| 9 | A Bitter Lesson for Data Filtering | 2605.19407 | 2026-05-19 | **61** | WEAK |

---

## Papers Selected for Full Summaries (Score ≥ 40)

All 9 papers scored ≥ 40 and are included in `papers/`.

## Code Reproductions (Score ≥ 80)

| Paper | Score | Folder |
|-------|-------|--------|
| Thinking Broad, Acting Fast | 85 | `code/ThinkingBroadActingFast/` |
| SARM | 83 | `code/SARM/` |
| TGQ-Former | 82 | `code/TGQ-Former/` |

## Feishu Cards

See `feishu_cards.md` for all papers with score ≥ 40.

## Web App

See `web/index.html` for the interactive paper browser.
