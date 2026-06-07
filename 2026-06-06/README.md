# Daily AI Paper Report — 2026-06-06 (GMT+8)

> **Domain**: E-commerce content ecosystem & influencer (达人) governance  
> **Emphasis**: LLM / VLM / MLLM, captioning, vector embeddings, clustering & similarity, agents, RAG, distillation, violation detection, content governance, data quality, data cleaning, large-scale data labeling  
> **Scheduled run**: 08:30 GMT+8

---

## Source Coverage

| Source | Category | Attempted | HTTP Status / Error | Candidates Yielded |
|--------|----------|-----------|--------------------|--------------------|
| arxiv cs.CL/new | aggregator | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| arxiv cs.CV/new | aggregator | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| arxiv cs.IR/new | aggregator | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| arxiv cs.MM/new | aggregator | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| arxiv cs.LG/new | aggregator | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| arxiv cs.AI/new | aggregator | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| HuggingFace papers/date/2026-06-06 | aggregator | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| HuggingFace papers/trending | aggregator | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| Google DeepMind blog | blog | yes | 403 Forbidden | 0 |
| Google Research blog | blog | no | not attempted (DeepMind already 403) | 0 |
| Meta AI Research | blog | yes | 403 Forbidden | 0 |
| Anthropic News | blog | yes | 403 Forbidden | 0 |
| ByteDance Seed Research | blog | yes | 403 Forbidden | 0 (found via WebSearch) |
| Meituan Tech | blog | yes | 403 Forbidden | 0 |
| Qwen Blog | blog | yes | 403 Forbidden | 0 |
| 量子位 QbitAI | wechat | yes | 403 Forbidden | 0 (fallback WebSearch, no June 6 hits) |
| Semantic Scholar Graph API | aggregator | yes | 403 Forbidden | 0 |
| WebSearch fallback: 机器之心 jiqizhixin | wechat | yes | 200 (WebSearch) | 0 (no June 6 papers found) |
| WebSearch fallback: 新智元 | wechat | yes | 200 (WebSearch) | 0 (no June 6 papers found) |
| WebSearch fallback: OpenAI News | blog | yes | 200 (WebSearch) | 0 (no matching results) |
| WebSearch fallback: DeepSeek | blog | yes | 200 (WebSearch) | 0 (no June 2026 new paper) |
| WebSearch fallback: Xiaohongshu/RedTech | wechat | yes | 200 (WebSearch) | 0 |
| WebSearch fallback: Kuaishou Tech | wechat | yes | 200 (WebSearch) | 0 (KuaiMod found via search, dated Apr 2025) |
| WebSearch fallback: Tencent Hunyuan | wechat | yes | 200 (WebSearch) | 0 |
| OpenReview ICLR/NeurIPS 2026 | conf | no | not yet active | 0 |
| arXiv 2606.xxxxx WebSearch (primary discovery) | aggregator | yes | 200 (WebSearch) | 8 candidates → 3 qualifying |

**Note**: arXiv HTML/PDF pages all returned 403 in this environment. All paper content was retrieved via `WebSearch` + follow-up `WebSearch` for abstract details. The Semantic Scholar API also returned 403. Paper details confirmed through multiple cross-validated searches.

---

## Papers Selected

| # | Paper | arXiv ID | Score | Bucket | Code |
|---|-------|----------|-------|--------|------|
| 1 | UNIVID: Unified VLM for Video Moderation | 2606.05748 | **87** | STRONG | `code/UNIVID/` |
| 2 | QueryAgent-R1: E-commerce Query Recommendation | 2606.05671 | **80** | STRONG | `code/QueryAgent-R1/` |
| 3 | OCL: Organizational Control Layer for LLM Agents | 2606.04306 | **70** | STRONG | — |

---

## Paper Summaries

### 1. UNIVID (Score: 87) — `papers/univid.md`

**ByteDance** introduces UNIVID, a Unified Vision-Language Model for large-scale video content moderation. Instead of thousands of policy-specific classifiers, a single VLM backbone generates structured, policy-aware captions that serve as an interpretable intermediate for downstream moderation decisions. The cascaded pipeline (Risk Filter → Moderation Actor with UNIVID-Lite + UNIVID-RAG → Trend Governance with embedding-based clustering) reduces violation leakage by **42.7%** and overkill rate by **37.0%** while retiring 1,000+ legacy models.

### 2. QueryAgent-R1 (Score: 80) — `papers/queryagent_r1.md`

**Alibaba International** proposes QueryAgent-R1, a memory-augmented RL agent that bridges query generation and product retrieval for e-commerce query recommendation. A chain-of-retrieval optimization loop validates generated queries against live inventory and a consistency reward jointly optimizes CTR and CVR. Online A/B tests yield **+2.9% CTR** and **+3.1% CVR** at production scale.

### 3. OCL (Score: 70) — `papers/ocl.md`

Multi-institution researchers introduce the Organizational Control Layer (OCL), a model-agnostic governance infrastructure for LLM agents. OCL intercepts proposed actions before execution and applies policy enforcement + escalation without modifying the underlying LLM. Unsafe executions drop from **88% to near-zero**, valid success rises from **12% to 96%** across GPT-5.4, Gemini-3.1, and Qwen-3.5 backends.
