# 📄 AI Paper Daily Inspection — 2026-05-21 (GMT+8)

**Domain:** E-commerce content ecosystem & influencer (达人) governance  
**Emphasis:** LLM / VLM / MLLM · captioning · vector embeddings · clustering & similarity · agents · RAG · distillation · violation detection · content governance · data quality

---

## Source Coverage Table

| Source | Category | Attempted | HTTP Status / Error | Candidates Yielded |
|--------|----------|-----------|--------------------|--------------------|
| arXiv cs.CL/new | aggregator | yes | 403 Forbidden | 0 (WebSearch fallback: 4) |
| arXiv cs.CV/new | aggregator | yes | 403 Forbidden | 0 (WebSearch fallback: 5) |
| arXiv cs.IR/new | aggregator | yes | 403 Forbidden | 0 (WebSearch fallback: 3) |
| arXiv cs.MM/new | aggregator | yes | 403 Forbidden | 0 (WebSearch fallback: 2) |
| arXiv cs.LG/new | aggregator | yes | 403 Forbidden | 0 (WebSearch fallback: 3) |
| arXiv cs.AI/new | aggregator | yes | 403 Forbidden | 0 (WebSearch fallback: 2) |
| HuggingFace Daily Papers 2026-05-21 | aggregator | yes | 403 Forbidden | 0 (WebSearch fallback) |
| HuggingFace Trending | aggregator | yes | 403 Forbidden | 0 |
| Semantic Scholar API (2026-05-21 window) | aggregator | yes | 403 Forbidden | 0 |
| Google DeepMind Blog | blog | yes | 403 Forbidden | 0 |
| Google Research Blog | blog | yes | 403 Forbidden | 0 |
| Meta AI Research | blog | yes | 403 Forbidden | 0 |
| Anthropic News | blog | yes | 403 Forbidden | 0 |
| ByteDance Seed Research | blog | yes | 403 Forbidden | 0 |
| Meituan Tech Blog | blog | yes | 403 Forbidden | 0 |
| Qwen Blog | blog | yes | 403 Forbidden | 0 |
| 量子位 QbitAI (qbitai.com) | wechat/web | yes | 403 Forbidden | 0 (WebSearch fallback: 3) |
| 机器之心 jiqizhixin.com | wechat | yes | WebSearch fallback | 0 relevant |
| 新智元 | wechat | yes | WebSearch fallback | 0 relevant |
| OpenAI News (WebSearch) | blog | yes | WebSearch fallback | 0 relevant |
| DeepSeek (WebSearch) | blog | yes | WebSearch fallback | 0 relevant |
| Xiaohongshu/RedTech (WebSearch) | wechat | yes | WebSearch fallback | 0 relevant |
| Kuaishou Tech (WebSearch) | wechat | yes | WebSearch fallback | 0 relevant |
| Tencent Hunyuan (WebSearch) | wechat | yes | WebSearch fallback | 0 relevant |
| OpenReview ICLR/NeurIPS 2026 | conf | no | not yet active | 0 |
| WebSearch (arXiv 2605.16xxx–2605.20xxx window) | aggregator | yes | 200 OK | **18 candidates** |

> **Network note:** The sandbox network returned HTTP 403 for all arXiv, HuggingFace, lab blog, and Semantic Scholar direct URLs. All discovery was conducted via WebSearch fallback. Papers with arXiv IDs 2605.16xxx–2605.20xxx (submitted ≈ May 16–19, 2026) are the primary candidates for the May 21 GMT+8 arXiv listing (arXiv publishes ~2 business days after submission). IDs 2605.01xxx–2605.15xxx are catch-up candidates found via search.

---

## Selected Papers — Scored

### STRONG (Directly E-commerce / Content Governance / Influencer)

| # | Title | arXiv ID | Submission | Score | Bucket |
|---|-------|----------|-----------|-------|--------|
| 1 | TGQ-Former: Text-Guided Visual Representation Learning for Robust Multimodal E-Commerce Recommendation | 2605.17366 | 2026-05-17 | **81** | STRONG |
| 2 | Valley3: Scaling Omni Foundation Models for E-commerce | 2605.01278 | 2026-05-02 | **80** | STRONG |
| 3 | PluRule: A Benchmark for Moderating Pluralistic Communities on Social Media | 2605.17187 | 2026-05-16 | **65** | STRONG |
| 4 | GRE-MC: Robust Multimodal Recommendation via Graph Retrieval-Enhanced Modality Completion | 2605.00670 | 2026-05-01 | **62** | STRONG |

### Notable Recent Papers (outside strict May 21 window, included for domain reference)

| # | Title | arXiv ID | Submission | Score | Note |
|---|-------|----------|-----------|-------|------|
| 5 | SARM: LLM-Augmented Semantic Anchor for End-to-End Live-Streaming Ranking | 2602.09401 | 2026-02-10 | **85** | Feb 2026; already indexed earlier |
| 6 | Deja Vu in Plots: CS-VAR for Live Streaming Risk Assessment | 2601.16027 | 2026-01-22 | **76** | Jan 2026; already indexed earlier |

> **De-duplication:** RuleSafe-VL (2605.07760) was already covered in the 2026-05-14 daily run and is excluded here. VLM as Policy (2504.14904) and Adapting VLMs for E-commerce (2602.11733) were covered in 2026-05-10 run.

---

## Code Reproductions (Score ≥ 80)

| Paper | Model Name | Folder | Code Available? |
|-------|-----------|--------|----------------|
| TGQ-Former (2605.17366) | `TGQFormer` | `code/TGQFormer/` | No public code → reproduced |
| Valley3 (2605.01278) | `Valley3` | `code/Valley3/` | No public code → reproduced |

---

## Paper Files

- `papers/tgq_former_ecom_rec.md` — TGQ-Former, score 81
- `papers/valley3_ecom_omni.md` — Valley3, score 80
- `papers/plurule_moderation_benchmark.md` — PluRule, score 65
- `papers/gre_mc_multimodal_rec.md` — GRE-MC, score 62
- `papers/sarm_livestream_ranking.md` — SARM, score 85 (reference, Jan 2026)
- `papers/csvar_livestream_risk.md` — CS-VAR, score 76 (reference, Jan 2026)
