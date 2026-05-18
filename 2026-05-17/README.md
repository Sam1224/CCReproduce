# Daily AI Paper Inspection — 2026-05-17 (GMT+8)

> **Scheduled run**: 08:30 GMT+8, 2026-05-18  
> **Window**: Papers indexed on 2026-05-17 (GMT+8)  
> **Domain**: E-commerce content ecosystem & influencer (达人) governance  
> **Note**: 2026-05-17 is a **Sunday**. arXiv does not publish new listings on weekends. The most recent available papers are from the Friday listing (submissions through Thursday 2026-05-15). HuggingFace aggregators also returned 403. This report covers the most recent and relevant papers discoverable from the May 2026 window.

---

## Source Coverage

| Source | Category | Attempted | HTTP Status / Error | Candidates Yielded |
|--------|----------|-----------|--------------------|--------------------|
| arxiv cs.CL/new | aggregator | yes | 403 Forbidden | 0 (direct) |
| arxiv cs.CV/new | aggregator | yes | 403 Forbidden | 0 (direct) |
| arxiv cs.IR/new | aggregator | yes | 403 Forbidden | 0 (direct) |
| arxiv cs.MM/new | aggregator | yes | 403 Forbidden | 0 (direct) |
| arxiv cs.LG/new | aggregator | yes | 403 Forbidden | 0 (direct) |
| arxiv cs.AI/new | aggregator | yes | 403 Forbidden | 0 (direct) |
| arxiv cs.CL/current | aggregator | yes | 403 Forbidden | 0 (direct) |
| HuggingFace papers/date/2026-05-17 | aggregator | yes | 403 Forbidden | 0 (direct) |
| HuggingFace papers/trending | aggregator | yes | 403 Forbidden | 0 (direct) |
| Google DeepMind blog | blog | yes | 403 Forbidden | 0 |
| Google Research blog | blog | yes | 403 Forbidden | 0 |
| Meta AI Research | blog | yes | 403 Forbidden | 0 |
| Anthropic News | blog | yes | 403 Forbidden | 0 |
| ByteDance Seed Research | blog | yes | 403 Forbidden | 0 |
| Meituan Tech | blog | yes | 403 Forbidden | 0 |
| Qwen Blog | blog | yes | 403 Forbidden | 0 |
| 量子位 QbitAI | wechat | yes | 403 Forbidden | 0 (direct); fallback WebSearch yielded 3 |
| Semantic Scholar Graph API | api | yes | 403 Forbidden | 0 |
| 机器之心 jiqizhixin.com | wechat/fallback | yes | WebSearch fallback | 2 |
| 新智元 mp.weixin.qq.com | wechat/fallback | yes | WebSearch fallback | 1 |
| OpenAI News (search fallback) | blog/fallback | yes | WebSearch | 1 |
| DeepSeek (search fallback) | blog/fallback | yes | WebSearch | 0 |
| 小红书 REDtech (search fallback) | blog/fallback | yes | WebSearch | 1 |
| 快手 Kuaishou Tech (search fallback) | blog/fallback | yes | WebSearch | 1 |
| 腾讯混元 Hunyuan (search fallback) | blog/fallback | yes | WebSearch | 0 |
| arXiv WebSearch fallback (cs.CL/CV/IR topics) | aggregator/fallback | yes | WebSearch | 7 |
| OpenReview ICLR/NeurIPS 2026 | conf | no | not yet active | 0 |

**Summary**: 26 sources attempted, 27 total WebFetch/WebSearch calls. Direct WebFetch blocked (403) for arXiv, HuggingFace, and all lab blogs. WebSearch fallback successfully yielded candidates. Note: 2026-05-17 = Sunday; arXiv has no new listings on weekends.

---

## Picked Papers

| # | Paper | ArXiv ID | Bucket | Score |
|---|-------|----------|--------|-------|
| 1 | EVADE: Multimodal Benchmark for Evasive Content Detection in E-Commerce | 2505.17654 | **STRONG** | **88** |
| 2 | MoMoE: Mixture of Moderation Experts for AI-Assisted Online Governance | 2505.14483 | **STRONG** | **80** |
| 3 | GLiGuard: Schema-Conditioned Classification for LLM Safeguard | 2605.07982 | **STRONG** | **72** |
| 4 | GRE-MC: Robust Multimodal Recommendation via Graph Retrieval-Enhanced Modality Completion | 2605.00670 | **STRONG** | **70** |
| 5 | RAG-Enhanced LLMs for Dynamic Content Expiration Prediction in Web Search | 2605.13052 | **STRONG** | **65** |
| 6 | GLiNER Guard: Unified Encoder Family for Production LLM Safety | 2605.05277 | **STRONG** | **60** |
| 7 | Is Grep All You Need? How Agent Harnesses Reshape Agentic Search | 2605.15184 | **WEAK** | **52** |

---

## Paper Details

- [EVADE](papers/evade.md) — Score: 88
- [MoMoE](papers/momoe.md) — Score: 80
- [GLiGuard](papers/gliguard.md) — Score: 72
- [GRE-MC](papers/gre_mc.md) — Score: 70
- [RAG-Content-Expiry](papers/rag_content_expiry.md) — Score: 65
- [GLiNER-Guard](papers/gliner_guard.md) — Score: 60
- [Is-Grep-All-You-Need](papers/is_grep_all_you_need.md) — Score: 52

---

## Code Reproductions (Score ≥ 80)

- [`code/EVADE/`](code/EVADE/) — EVADE benchmark evaluation framework
- [`code/MoMoE/`](code/MoMoE/) — Mixture of Moderation Experts implementation
