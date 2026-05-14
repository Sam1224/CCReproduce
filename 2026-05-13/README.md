# 📄 AI Paper Daily Inspection — 2026-05-13 (GMT+8)

**Domain:** E-commerce content ecosystem & influencer governance  
**Emphasis:** LLM / VLM / MLLM · captioning · vector embeddings · clustering & similarity · agents · RAG · distillation · violation detection · content governance · data quality

---

## Source Coverage Table

| Source | Category | Attempted | HTTP Status / Error | Candidates Yielded |
|--------|----------|-----------|--------------------|--------------------|
| arXiv cs.CL/new (2026-05-13) | aggregator | yes | 403 Forbidden | 0 (web search fallback: ~6) |
| arXiv cs.CV/new (2026-05-13) | aggregator | yes | 403 Forbidden | 0 (web search fallback: ~4) |
| arXiv cs.IR/new (2026-05-13) | aggregator | yes | 403 Forbidden | 0 (web search fallback: ~3) |
| arXiv cs.MM/new (2026-05-13) | aggregator | yes | 403 Forbidden | 0 (web search fallback: ~2) |
| arXiv cs.LG/new (2026-05-13) | aggregator | yes | 403 Forbidden | 0 (web search fallback: ~3) |
| arXiv cs.AI/new (2026-05-13) | aggregator | yes | 403 Forbidden | 0 (web search fallback: ~4) |
| HuggingFace Daily Papers 2026-05-13 | aggregator | yes | 403 Forbidden | 0 |
| HuggingFace Trending Papers | aggregator | yes | 403 Forbidden | 0 |
| Semantic Scholar Graph API | aggregator | yes | 403 Forbidden | 0 |
| Google DeepMind Blog | blog | yes | 403 Forbidden | 0 |
| Google Research Blog | blog | yes | 403 Forbidden | 0 |
| Meta AI Research | blog | yes | 403 Forbidden | 0 |
| Anthropic News | blog | yes | 403 Forbidden | 0 |
| ByteDance Seed Research | blog | yes | 403 Forbidden | 0 (web search: 2) |
| Meituan Tech Blog | blog | yes | 403 Forbidden | 0 |
| Qwen Blog | blog | no | not yet active / deprioritized | 0 |
| 量子位 QbitAI | wechat | yes | 403 Forbidden | 0 (web search: 1 article about Baidu Create 2026) |
| 机器之心 jiqizhixin.com | wechat | yes | WebSearch fallback | 0 relevant candidates |
| 新智元 (WeChat) | wechat | yes | WebSearch fallback | 0 relevant candidates |
| OpenAI News | blog | yes | WebSearch fallback | 0 relevant candidates |
| DeepSeek | blog | yes | WebSearch fallback | 0 relevant candidates |
| Xiaohongshu / RedTech | wechat | yes | WebSearch fallback | 0 relevant candidates |
| Kuaishou Tech | wechat | yes | WebSearch fallback | 0 relevant candidates |
| Tencent Hunyuan | wechat | yes | WebSearch fallback | 0 relevant candidates |
| OpenReview ICLR / NeurIPS 2026 | conf | no | not yet active | 0 |
| WebSearch Tier-3 fallback (all topics) | aggregator | yes | 200 OK | **26 candidates** |

> **Network note:** The sandbox network blocks all external HTTP/HTTPS endpoints except GitHub, returning 403 Forbidden. All paper discovery was conducted via WebSearch fallback. Candidates are from arXiv IDs 2605.01xxx–2605.09xxx (papers submitted May 1–9, 2026). GLiGuard (2605.07982) was confirmed indexed on May 13, 2026 via MarkTechPost article timestamped that date.

---

## Selected Papers — Scored

### STRONG (Directly Ecom / Content Governance / Influencer)

| # | Title | arXiv ID | Score | Bucket |
|---|-------|----------|-------|--------|
| 1 | A Case-Driven Multi-Agent Framework for E-Commerce Search Relevance | 2605.05991 | **81** | STRONG |
| 2 | ARGUS: Policy-Adaptive Ad Governance via Evolving Reinforcement with Adversarial Umpiring | 2605.02200 | **80** | STRONG |
| 3 | Valley3: Scaling Omni Foundation Models for E-commerce | 2605.01278 | **76** | STRONG |
| 4 | GLiGuard: Schema-Conditioned Classification for LLM Safeguard | 2605.07982 | **74** | STRONG |

### WEAK (High-Impact LLM/VLM, Transferable to Ecom/Governance)

| # | Title | arXiv ID | Score | Bucket |
|---|-------|----------|-------|--------|
| 5 | MELD: Multi-Task Equilibrated Learning Detector for AI-Generated Text | 2605.06903 | **58** | WEAK |

---

## Code Reproductions (Score ≥ 80)

| Paper | Model Name | Folder |
|-------|-----------|--------|
| A Case-Driven Multi-Agent Framework for E-Commerce Search Relevance | `CaseDrivenMASearch` | `code/CaseDrivenMASearch/` |
| ARGUS: Policy-Adaptive Ad Governance | `ARGUS` | `code/ARGUS/` |

---

## Paper Details Links

- [A Case-Driven Multi-Agent Framework](https://arxiv.org/abs/2605.05991) → `papers/case_driven_ma_ecom_search.md`
- [ARGUS Ad Governance](https://arxiv.org/abs/2605.02200) → `papers/argus_ad_governance.md`
- [Valley3 E-commerce Omni MLLM](https://arxiv.org/abs/2605.01278) → `papers/valley3_ecom_omni.md`
- [GLiGuard LLM Safeguard](https://arxiv.org/abs/2605.07982) → `papers/gliguard_llm_safeguard.md`
- [MELD AIGC Text Detection](https://arxiv.org/abs/2605.06903) → `papers/meld_aigc_text_detection.md`
