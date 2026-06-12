# Daily AI Paper Inspection — 2026-06-11 (GMT+8)

**Domain:** E-commerce content ecosystem & influencer (达人) governance  
**Run time:** 2026-06-12 08:30 GMT+8  
**Discovered by:** Claude Code scheduled inspection  

---

## Source Coverage

| source | category | attempted | http_status_or_error | candidates_yielded |
|--------|----------|-----------|----------------------|--------------------|
| arxiv cs.CL/new (WebFetch) | aggregator | yes | 403 Forbidden | 0 |
| arxiv cs.CV/new (WebFetch) | aggregator | yes | 403 Forbidden | 0 |
| arxiv cs.IR/new (WebFetch) | aggregator | yes | 403 Forbidden | 0 |
| arxiv cs.MM/new (WebFetch) | aggregator | yes | 403 Forbidden | 0 |
| arxiv cs.LG/new (WebFetch) | aggregator | yes | 403 Forbidden | 0 |
| arxiv cs.AI/new (WebFetch) | aggregator | yes | 403 Forbidden | 0 |
| arxiv 2606 series (WebSearch fallback) | aggregator | yes | 200 OK | 9 |
| HuggingFace papers/date/2026-06-11 | aggregator | yes | 403 Forbidden | 0 |
| HuggingFace trending | aggregator | yes | 403 Forbidden | 0 |
| HuggingFace (WebSearch fallback) | aggregator | yes | 200 OK | 2 |
| Google DeepMind blog | blog | yes | 403 Forbidden | 0 |
| Google Research blog | blog | yes | 403 Forbidden | 0 |
| Meta AI Research | blog | yes | 403 Forbidden | 0 |
| Anthropic News | blog | yes | 403 Forbidden | 0 |
| ByteDance Seed Research | blog | yes | 403 Forbidden | 0 |
| Meituan Tech | blog | yes | 403 Forbidden | 0 |
| Qwen Blog | blog | yes | 403 Forbidden | 0 |
| 量子位 QbitAI (WebFetch direct) | wechat | yes | 403 Forbidden | 0 |
| 量子位 QbitAI (WebSearch fallback) | wechat | yes | 200 OK | 3 |
| Semantic Scholar Graph API | aggregator | yes | 403 Forbidden | 0 |
| 机器之心 jiqizhixin (WebSearch Tier-3) | wechat | yes | 200 OK | 1 |
| 新智元 (WebSearch Tier-3) | wechat | yes | 200 OK | 1 |
| OpenAI News (WebSearch Tier-3) | blog | yes | 200 OK | 1 |
| DeepSeek (WebSearch Tier-3) | blog | yes | 200 OK | 0 |
| Xiaohongshu / RedTech (WebSearch Tier-3) | wechat | yes | 200 OK | 0 |
| Kuaishou Tech (WebSearch Tier-3) | blog | yes | 200 OK | 1 |
| Tencent Hunyuan (WebSearch Tier-3) | blog | yes | 200 OK | 0 |
| OpenReview ICLR/NeurIPS 2026 | conf | no | not yet active | 0 |

**Note:** arXiv direct pages and all lab blogs blocked with HTTP 403 in this sandbox. Discovery performed entirely via WebSearch fallback, targeting 2606.XXXXX arXiv IDs (June 2026). Paper submission dates verified from search snippet metadata where available.

---

## Picked Papers

| # | Title | ArXiv ID | Score | Bucket | Submission Date |
|---|-------|----------|-------|--------|----------------|
| 1 | UNIVID: Unified Vision-Language Model for Video Moderation | 2606.05748 | **84** | STRONG | Jun 4, 2026 |
| 2 | QueryAgent-R1: E-Commerce Query Recommendation | 2606.05671 | **82** | STRONG | Jun 4, 2026 |
| 3 | Unintended Consequences of Recommender System Interventions | 2606.08265 | **71** | STRONG | Jun 6, 2026 |
| 4 | EVID-Bench: Search-Grounded Video Misinformation Detection | 2606.04098 | **70** | STRONG | Jun 3, 2026 |
| 5 | Active Learning with Foundation Model Priors | 2606.07630 | **64** | WEAK | May 30, 2026 |
| 6 | Mitigating Perceptual Judgment Bias in MLLM-as-Judge | 2606.02578 | **60** | WEAK | Jun 2, 2026 |
| 7 | Reroute, Don't Remove: Recoverable Visual Token Routing | 2606.12412 | **59** | WEAK | Jun 11, 2026 |
| 8 | Context-Driven Incremental Compression for Dialogue | 2606.12411 | **58** | WEAK | Jun 11, 2026 |
| 9 | InSemRAG: Efficient RAG with Intent-Aware Retrieval | 2606.01240 | **56** | WEAK | Jun 1, 2026 |

Papers with **Score ≥ 80**: UNIVID, QueryAgent-R1 → code reproductions at `code/UNIVID/` and `code/QueryAgent-R1/`

---

## Paper Summaries (Index)

1. [UNIVID](papers/univid.md) — ByteDance, score 84 — Video content moderation VLM
2. [QueryAgent-R1](papers/queryagent_r1.md) — Alibaba International, score 82 — E-commerce query recommendation
3. [Recommender Interventions](papers/recommender_interventions.md) — Washington University, score 71 — Platform governance
4. [EVID-Bench](papers/evid_bench.md) — score 70 — Video misinformation benchmark
5. [Active Learning FM Priors](papers/active_learning_fm_priors.md) — score 64 — Data labeling quality
6. [Perceptual Judgment Bias](papers/perceptual_judgment_bias.md) — KAIST (ICML 2026), score 60 — MLLM evaluation
7. [Reroute VLM Token Routing](papers/reroute_vlm.md) — score 59 — VLM efficiency (Jun 11)
8. [C-DIC Dialogue Compression](papers/cdic_dialogue.md) — ICML 2026, score 58 — LLM dialogue efficiency (Jun 11)
9. [InSemRAG](papers/insemrag.md) — score 56 — RAG efficiency
