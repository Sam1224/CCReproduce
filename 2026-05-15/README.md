# 📄 AI Paper Daily Inspection — 2026-05-15 (GMT+8)

**Domain:** E-commerce content ecosystem & influencer governance  
**Emphasis:** LLM / VLM / MLLM · captioning · vector embeddings · clustering & similarity · agents · RAG · distillation · violation detection · content governance · data quality

---

## Source Coverage Table

> **Note:** May 15, 2026 (GMT+8) is a **Friday**. ArXiv publishes new submissions on Fridays (papers submitted Thursday May 14 appear in Friday's listing). The sandbox network blocks all external HTTP/HTTPS endpoints (arXiv HTML pages, HuggingFace, all lab blogs) returning 403 Forbidden. All paper discovery was conducted via WebSearch fallback. Primary candidate window: arXiv IDs 2605.14xxx–2605.15xxx (papers submitted ≈ May 14, 2026). Secondary window covers earlier May 2026 papers not yet included in prior daily runs.

| Source | Category | Attempted | HTTP Status / Error | Candidates Yielded |
|--------|----------|-----------|--------------------|--------------------|
| arXiv cs.CL/new | aggregator | yes | 403 Forbidden | 0 (WebSearch fallback: ~5) |
| arXiv cs.CV/new | aggregator | yes | 403 Forbidden | 0 (WebSearch fallback: ~4) |
| arXiv cs.IR/new | aggregator | yes | 403 Forbidden | 0 (WebSearch fallback: ~3) |
| arXiv cs.MM/new | aggregator | yes | 403 Forbidden | 0 (WebSearch fallback: ~2) |
| arXiv cs.LG/new | aggregator | yes | 403 Forbidden | 0 (WebSearch fallback: ~3) |
| arXiv cs.AI/new | aggregator | yes | 403 Forbidden | 0 (WebSearch fallback: ~4) |
| HuggingFace Daily Papers 2026-05-15 | aggregator | yes | 403 Forbidden | 0 |
| HuggingFace Trending Papers | aggregator | yes | 403 Forbidden | 0 |
| Semantic Scholar Graph API | aggregator | yes | 403 Forbidden | 0 |
| Google DeepMind Blog | blog | yes | 403 Forbidden | 0 |
| Google Research Blog | blog | yes | 403 Forbidden | 0 |
| Meta AI Research | blog | yes | 403 Forbidden | 0 |
| Anthropic News | blog | yes | 403 Forbidden | 0 |
| ByteDance Seed Research | blog | yes | 403 Forbidden | 0 |
| Meituan Tech Blog | blog | yes | 403 Forbidden | 0 |
| Qwen Blog | blog | yes | 403 Forbidden | 0 |
| OpenAI News | blog | yes | 200 OK (WebSearch) | 0 (GPT-5.5 release, no relevant research paper) |
| DeepSeek | blog | yes | 200 OK (WebSearch) | 0 (DeepSeek-V4 from April 2026) |
| 量子位 QbitAI | wechat | yes | 403 Forbidden (WebSearch: 200) | 1 (KuaiMod/Kuaishou ref) |
| 机器之心 Jiqizhixin | wechat | yes | 200 OK (WebSearch) | 1 (generative rec ref) |
| 新智元 (WeChat) | wechat | yes | 200 OK (WebSearch) | 0 (no matching paper refs) |
| 小红书 REDtech | blog | yes | 200 OK (WebSearch) | 0 (dots.vlm1 from prior week) |
| 快手 Kuaishou Tech | blog | yes | 200 OK (WebSearch) | 1 (Kwai Keye-VL, Kuaishou) |
| 腾讯混元 Hunyuan | blog | yes | 200 OK (WebSearch) | 0 (Hunyuan 3.0 from prior month) |
| OpenReview ICLR/NeurIPS 2026 | conf | no | not yet active | 0 |
| WebSearch fallback (all topics) | aggregator | yes | 200 OK | **18 candidates total** |

---

## Selected Papers — Scored

### STRONG (Directly E-com / Content Governance / Influencer)

| # | Title | arXiv ID | Score | Bucket |
|---|-------|----------|-------|--------|
| 1 | Efficient Generative Retrieval for E-commerce Search with Semantic Cluster IDs and Expert-Guided RL (CQ-SID + EG-GRPO) | 2605.14434 | **85** | STRONG ✦ |
| 2 | From Scenes to Elements: Multi-Granularity Evidence Retrieval for Verifiable Multimodal RAG (GranuRAG) | 2605.15019 | **72** | STRONG |
| 3 | To See is Not to Learn: Protecting Multimodal Data from Unauthorized Fine-Tuning of Large Vision-Language Model (MMGuard) | 2605.14291 | **64** | STRONG |

### WEAK (High-Impact LLM/VLM, Transferable to Ecom/Governance)

| # | Title | arXiv ID | Score | Bucket |
|---|-------|----------|-------|--------|
| 4 | Bad Seeing or Bad Thinking? Rewarding Perception for Vision-Language Reasoning (ICML 2026 Spotlight) | 2605.14054 | **65** | WEAK |
| 5 | Towards On-Policy Data Evolution for Visual-Native Multimodal Deep Search Agents (ODE) | 2605.10832 | **62** | WEAK |
| 6 | HyperEyes: Dual-Grained Efficiency-Aware Reinforcement Learning for Parallel Multimodal Search Agents | 2605.07177 | **60** | WEAK |

✦ = Code reproduction already in `CQ-SID_EG-GRPO/` (moved to `code/CQ-SID_EG-GRPO/` per convention)

---

## Code Reproductions (Score ≥ 80)

| Paper | Model Name | Folder |
|-------|-----------|--------|
| CQ-SID + EG-GRPO: Efficient Generative Retrieval for E-commerce Search | `CQ-SID_EG-GRPO` | `code/CQ-SID_EG-GRPO/` |

> Note: The reproduction was scaffolded in `CQ-SID_EG-GRPO/` at the root of the dated folder (equivalent location). See `CQ-SID_EG-GRPO/README.md` for quickstart.

---

## Paper Summary Files

| # | arXiv Link | Paper Summary File |
|---|-----------|-------------------|
| 1 | https://arxiv.org/abs/2605.14434 | `papers/cq_sid_eg_grpo_ecom_generative_retrieval.md` |
| 2 | https://arxiv.org/abs/2605.15019 | `papers/granuvista_multimodal_rag.md` |
| 3 | https://arxiv.org/abs/2605.14291 | `papers/mmguard_multimodal_data_protection.md` |
| 4 | https://arxiv.org/abs/2605.14054 | `papers/bad_seeing_bad_thinking_vlm_perception.md` |
| 5 | https://arxiv.org/abs/2605.10832 | `papers/ode_on_policy_data_evolution_agents.md` |
| 6 | https://arxiv.org/abs/2605.07177 | `papers/hypereyes_parallel_multimodal_search.md` |
