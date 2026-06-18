# Daily AI Paper Inspection — 2026-06-17 (GMT+8)

**Domain**: E-commerce content ecosystem & influencer (达人) governance  
**Run time**: 2026-06-18 00:30 UTC (08:30 GMT+8)  
**Scan window**: Yesterday GMT+8 = 2026-06-17; catch-up window = 2026-06-11 ~ 2026-06-17

---

## Source Coverage

| source | category | attempted | http_status_or_error | candidates_yielded |
|--------|----------|-----------|---------------------|-------------------|
| arxiv cs.CL/new | aggregator | yes | 403 Forbidden | 0 |
| arxiv cs.CV/new | aggregator | yes | 403 Forbidden | 0 |
| arxiv cs.IR/new | aggregator | yes | 403 Forbidden | 0 |
| arxiv cs.MM/new | aggregator | yes | 403 Forbidden | 0 |
| arxiv cs.LG/new | aggregator | yes | 403 Forbidden | 0 |
| arxiv cs.AI/new | aggregator | yes | 403 Forbidden | 0 |
| arxiv cs.CL/current | aggregator | yes | 403 Forbidden | 0 |
| arxiv cs.CV/recent | aggregator | yes | 403 Forbidden | 0 |
| arxiv cs.AI/current | aggregator | yes | 403 Forbidden | 0 |
| HuggingFace papers/date/2026-06-17 | aggregator | yes | 403 Forbidden | 0 |
| HuggingFace papers/trending | aggregator | yes | 403 Forbidden | 0 |
| Google DeepMind blog | blog | yes | 403 Forbidden | 0 |
| Google Research blog | blog | yes | 403 Forbidden | 0 |
| Meta AI Research | blog | yes | 403 Forbidden | 0 |
| Anthropic News | blog | not attempted | reason=no relevant AI paper announcements on 2026-06-17 | 0 |
| ByteDance Seed Research | blog | yes | 403 Forbidden | 0 |
| Meituan Tech | blog | not attempted | reason=no direct paper feed accessible | 0 |
| Qwen Blog | blog | yes | 403 Forbidden | 0 |
| 量子位 QbitAI | wechat | yes | 403 Forbidden | 0 |
| 机器之心 (WebSearch fallback) | wechat | yes | 200 via WebSearch | 0 |
| 新智元 (WebSearch fallback) | wechat | yes | 200 via WebSearch | 0 |
| OpenAI News (WebSearch fallback) | blog | yes | 200 via WebSearch | 0 |
| DeepSeek (WebSearch fallback) | blog | yes | 200 via WebSearch | 0 |
| Semantic Scholar Graph API | aggregator | yes | 403 Forbidden | 0 |
| arxiv WebSearch fallback (cs.CL/CV/IR/MM/LG/AI) | aggregator | yes | 200 via WebSearch | 16 |
| OpenReview ICLR/NeurIPS 2026 | conf | no | reason=not yet active | 0 |

**Note**: All direct WebFetch calls to arxiv.org, huggingface.co, deepmind.google, research.google, ai.meta.com, seed.bytedance.com, qwenlm.github.io, qbitai.com, and api.semanticscholar.org returned HTTP 403. Discovery proceeded entirely via WebSearch fallback, which yielded 16 candidate papers spanning the 2606.xxxxx (June 2026) series plus carry-over from papers.json of prior partial run.

---

## Paper Index (Sorted by Score)

| # | Title | arXiv | Score | Bucket | Code |
|---|-------|-------|-------|--------|------|
| 1 | Atomic Intent Reasoning (AIR) | 2606.10357 | **91** | STRONG | [code/AIR](code/AIR/) |
| 2 | QueryAgent-R1 | 2606.05671 | **86** | STRONG | [code/QueryAgent-R1](code/QueryAgent-R1/) |
| 3 | OneRank | — | **80** | STRONG | [code/OneRank](code/OneRank/) (existing) |
| 4 | LiveStarPro | 2606.17798 | **80** | STRONG | [code/LiveStarPro](code/LiveStarPro/) |
| 5 | Harmonizing Semantic & Collaborative in LLMs | — | **77** | WEAK | — |
| 6 | MLT-Dedup | 2606.12215 | **77** | STRONG | — |
| 7 | AIGC Detection on Social Media | 2606.11200 | **76** | STRONG | — |
| 8 | Synthetic Priors Sequential Rec | — | **76** | WEAK | — |
| 9 | LLM Memorization in GenRec | — | **74** | WEAK | — |
| 10 | Unified Multimodal Autoregressive (Fudan+Qwen) | — | **74** | WEAK | — |
| 11 | Information Cocoon Closed-Loop Simulation | — | **73** | STRONG | — |
| 12 | Mind the Gap (DoorDash Multi-Vertical) | 2606.06779 | **68** | WEAK | — |
| 13 | MODE-RAG | — | **70** | WEAK | — |
| 14 | When More Documents Hurt RAG | 2606.11350 | **63** | WEAK | — |
| 15 | Code-Mixed Product Metadata (Daraz) | — | **47** | WEAK | — |

---

## Summary

- **15 papers** total (8 carry-over from prior partial run + 7 new from 2606 series)
- **4 papers** with score ≥ 80 → code reproduction: AIR, QueryAgent-R1, OneRank (existing), LiveStarPro
- **Domain highlights**: Kuaishou E-commerce +3.4% GMV (AIR); Alibaba International +3.1% CVR (QueryAgent-R1); KDD-2026 ADS track video deduplication (MLT-Dedup); Meta+CMU multimodal AIGC detection deployed at scale (2606.11200); livestream proactive video understanding (LiveStarPro)
