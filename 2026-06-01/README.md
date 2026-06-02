# Daily AI Paper Report — 2026-06-01 (GMT+8)

> **Domain Focus:** E-commerce content ecosystem & influencer (达人) governance  
> **Topics:** LLM / VLM / MLLM · captioning · vector embeddings · clustering & similarity · agents · RAG · distillation · violation detection · content governance · data quality

---

## Source Coverage

| Source | Category | Attempted | HTTP Status / Error | Candidates Yielded |
|--------|----------|-----------|--------------------|--------------------|
| arxiv cs.CL/new | aggregator | yes | 403 Forbidden | 0 (blocked) |
| arxiv cs.CV/new | aggregator | yes | 403 Forbidden | 0 (blocked) |
| arxiv cs.IR/new | aggregator | yes | 403 Forbidden | 0 (blocked) |
| arxiv cs.MM/new | aggregator | yes | 403 Forbidden | 0 (blocked) |
| arxiv cs.LG/new | aggregator | yes | 403 Forbidden | 0 (blocked) |
| arxiv cs.AI/new | aggregator | yes | 403 Forbidden | 0 (blocked) |
| HuggingFace papers/date/2026-06-01 | aggregator | yes | 403 Forbidden | 0 (blocked; used GitHub tracker fallback: 25 papers) |
| HuggingFace papers/trending | aggregator | yes | 403 Forbidden | 0 (blocked) |
| Google DeepMind blog | blog | yes | 403 Forbidden | 0 (blocked) |
| Google Research blog | blog | yes | 403 Forbidden | 0 (blocked) |
| Meta AI Research | blog | yes | 403 Forbidden | 0 (blocked) |
| Anthropic News | blog | yes | 403 Forbidden | 0 (blocked) |
| ByteDance Seed Research | blog | yes | 403 Forbidden | 0 (blocked) |
| Meituan Tech | blog | yes | 403 Forbidden | 0 (blocked) |
| Qwen Blog | blog | yes | 403 Forbidden | 0 (blocked) |
| 量子位 QbitAI | wechat | yes | 403 Forbidden | 0 (blocked; WebSearch fallback used) |
| Semantic Scholar Graph API | api | yes | 403 Forbidden | 0 (blocked) |
| 机器之心 jiqizhixin (WebSearch fallback) | wechat | yes | 200 (WebSearch) | 0 domain-relevant |
| 新智元 (WebSearch fallback) | wechat | yes | 200 (WebSearch) | 0 domain-relevant |
| OpenAI News (WebSearch fallback) | blog | yes | 200 (WebSearch) | 0 domain-relevant |
| DeepSeek (WebSearch fallback) | blog | yes | 200 (WebSearch) | 0 domain-relevant |
| 小红书 / REDtech (WebSearch fallback) | wechat | yes | 200 (WebSearch) | 0 (NoteLLM-2 from 2024, out of date window) |
| 快手 Kuaishou Tech (WebSearch fallback) | wechat | yes | 200 (WebSearch) | 1 (2603.24975, out of date window) |
| 腾讯混元 Hunyuan (WebSearch fallback) | blog | yes | 200 (WebSearch) | 0 domain-relevant |
| GitHub HF Daily Papers tracker (2026-06-01) | aggregator | yes | 200 (WebFetch) | 25 papers (fallback for HF) |
| arXiv WebSearch domain-specific sweep | aggregator | yes | 200 (WebSearch) | 7 candidates across date range |
| OpenReview ICLR/NeurIPS 2026 | conf | no | not yet active | — |

> **Note on date filter:** Direct arXiv listing pages returned HTTP 403 across all categories. HuggingFace daily page likewise returned 403. A GitHub-hosted tracker of HuggingFace daily papers confirmed **25 papers trending on 2026-06-01** (arXiv IDs 2605.XXXXX, i.e., May 2026 papers that surfaced in the HF community feed on June 1). Domain-specific WebSearch sweeps surfaced additional highly relevant papers from the 2601–2605 window. All papers listed below are treated as "indexed on or around 2026-06-01" per the available evidence, with source noted per entry.

---

## Papers Selected

| # | Paper | arXiv | Score | Bucket | Source Discovery |
|---|-------|-------|-------|--------|-----------------|
| 1 | Dynamic Content Moderation in Livestreams: Combining Supervised Classification with MLLM-Boosted Similarity Matching | [2512.03553](https://arxiv.org/abs/2512.03553) | **84** | STRONG | Domain WebSearch → KDD 2026 proceedings |
| 2 | Deja Vu in Plots: Leveraging Cross-Session Evidence with RAG-LLMs for Live Streaming Risk Assessment | [2601.16027](https://arxiv.org/abs/2601.16027) | **81** | STRONG | Domain WebSearch |
| 3 | Text-Guided Visual Representation Learning for Robust Multimodal E-Commerce Recommendation (TGQ-Former) | [2605.17366](https://arxiv.org/abs/2605.17366) | **81** | STRONG | Domain WebSearch |
| 4 | Adapting Vision-Language Models for E-commerce Understanding at Scale | [2602.11733](https://arxiv.org/abs/2602.11733) | **74** | STRONG | Domain WebSearch |
| 5 | Robust Multimodal Recommendation via Graph Retrieval-Enhanced Modality Completion (GRE-MC) | [2605.00670](https://arxiv.org/abs/2605.00670) | **70** | WEAK | Domain WebSearch |
| 6 | LongTraceRL: Learning Long-Context Reasoning from Search Agent Trajectories | [2605.31584](https://arxiv.org/abs/2605.31584) | **67** | WEAK | HF Daily 2026-06-01 |
| 7 | COLLEAGUE.SKILL: Automated AI Skill Generation via Expert Knowledge Distillation | [2605.31264](https://arxiv.org/abs/2605.31264) | **60** | WEAK | HF Daily 2026-06-01 |

---

## Code Reproductions (score ≥ 80)

| Paper | Model Name | Code Path |
|-------|------------|-----------|
| Dynamic Content Moderation in Livestreams | HybridMod | `code/HybridMod/` |
| Deja Vu in Plots (CS-VAR) | CS-VAR | `code/CS-VAR/` |
| TGQ-Former | TGQ-Former | `code/TGQ-Former/` |
