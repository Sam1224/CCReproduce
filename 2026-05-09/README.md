# 📄 AI Paper Daily Inspection — 2026-05-09 (GMT+8)

**Domain:** E-commerce content ecosystem & influencer governance  
**Emphasis:** LLM / VLM / MLLM · captioning · vector embeddings · clustering & similarity · agents · RAG · distillation · violation detection · content governance · data quality · large-scale data labeling

> **Backfill Note:** 2026-05-09 is a Saturday in GMT+8 (Friday May 8 ET). arXiv does not publish new announcements on weekends. The effective arXiv window is the **Friday May 8 listing** (papers submitted by Thursday May 7, 14:00 ET), corresponding to arXiv IDs ~2605.05xxx–2605.08xxx. All external HTTP/HTTPS hosts except GitHub return 403/ECONNREFUSED in this sandbox; all discovery was performed via WebFetch (mostly blocked) + WebSearch fallback.

---

## Source Coverage Table

| Source | Category | Attempted | HTTP Status / Error | Candidates Yielded |
|--------|----------|-----------|--------------------|--------------------|
| arXiv cs.CL/pastweek | aggregator | yes | 403 Forbidden | 0 (WebSearch fallback: ~12) |
| arXiv cs.CV/pastweek | aggregator | yes | 403 Forbidden | 0 (WebSearch fallback: ~8) |
| arXiv cs.IR/pastweek | aggregator | yes | 403 Forbidden | 0 (WebSearch fallback: ~5) |
| arXiv cs.MM/pastweek | aggregator | yes | 403 Forbidden | 0 |
| arXiv cs.LG/pastweek | aggregator | yes | 403 Forbidden | 0 (WebSearch fallback: ~4) |
| arXiv cs.AI/pastweek | aggregator | yes | 403 Forbidden | 0 (WebSearch fallback: ~6) |
| arXiv cs.CL/2026-05 (direct date) | aggregator | yes | 403 Forbidden | 0 |
| HuggingFace Daily Papers 2026-05-09 | aggregator | yes | 403 Forbidden | 0 |
| HuggingFace Trending Papers | aggregator | yes | 403 Forbidden | 0 |
| Semantic Scholar Graph API | aggregator | yes | 403 Forbidden | 0 |
| arXiv HTML abstracts (2605.xxxxx) | aggregator | yes | 403 Forbidden | 0 (WebSearch: 8 candidates) |
| Google DeepMind Blog | blog | yes | 403 Forbidden | 0 |
| Google Research Blog | blog | yes | 403 Forbidden | 0 |
| Meta AI Research | blog | yes | 403 Forbidden | 0 |
| Anthropic News | blog | yes | 403 Forbidden | 0 |
| ByteDance Seed Research | blog | yes | 403 Forbidden | 0 (WebSearch: 1 note) |
| Meituan Tech Blog | blog | yes | 403 Forbidden | 0 |
| Qwen Blog | blog | yes | 403 Forbidden | 0 |
| 量子位 QbitAI (qbitai.com) | wechat/blog | yes | 403 Forbidden | 1 (via WebSearch) |
| 机器之心 (jiqizhixin.com) | wechat | yes | WebSearch only | 0 domain-relevant |
| 新智元 (mp.weixin.qq.com) | wechat | yes | WebSearch only | 0 |
| OpenAI News/Index | blog | yes | WebSearch only | 0 |
| DeepSeek | blog | yes | WebSearch only | 0 |
| 小红书 / REDtech | blog | yes | WebSearch only | 0 |
| 快手 Tech / Kuaishou | blog | yes | WebSearch only | 0 |
| 腾讯混元 / Tencent Hunyuan | blog | yes | WebSearch only | 0 |
| OpenReview ICLR/NeurIPS 2026 | conf | no | not yet active | 0 |
| WebSearch fallback (all topics) | aggregator | yes | 200 OK | **28 candidates** |

> **Total candidates discovered:** 28 raw → 8 scored → 8 included in report.

---

## Selected Papers — Scored

Papers are from May 2026 (arXiv IDs 2605.xxxxx), indexed in the Friday May 8 / Saturday May 9 (GMT+8) window.

### STRONG (Directly Ecom / Content Governance / Influencer)

| # | Title | arXiv ID | Submitted | Score | Bucket |
|---|-------|----------|-----------|-------|--------|
| 1 | ARGUS: Policy-Adaptive Ad Governance via Evolving Reinforcement with Adversarial Umpiring | 2605.02200 | 2026-05-04 | **84** | STRONG |
| 2 | GLiGuard: Schema-Conditioned Classification for LLM Safeguard | 2605.07982 | 2026-05-08 | **80** | STRONG |
| 3 | DRIP-R: A Benchmark for Decision-Making and Reasoning Under Real-World Policy Ambiguity in the Retail Domain | 2605.07699 | 2026-05-08 | **76** | STRONG |
| 4 | Lightweight Stylistic Consistency Profiling: Robust Detection of LLM-Generated Textual Content for Multimedia Moderation | 2605.05950 | 2026-05-07 | **75** | STRONG |

### WEAK (High-Impact LLM/VLM Infrastructure, Transferable to Ecom/Governance)

| # | Title | arXiv ID | Submitted | Score | Bucket |
|---|-------|----------|-----------|-------|--------|
| 5 | Who Decides What Is Harmful? Content Moderation Policy Through A Multi-Agent Personalised Inference Framework | 2605.01416 | 2026-05-02 | **66** | WEAK |
| 6 | Multimodal Data Curation Through Ranked Retrieval | 2605.01163 | 2026-05-01 | **69** | WEAK |
| 7 | The Cost of Context: Mitigating Textual Bias in Multimodal Retrieval-Augmented Generation | 2605.05594 | 2026-05-07 | **64** | WEAK |
| 8 | Estimating the Black-box LLM Uncertainty with Distribution-Aligned Adversarial Distillation | 2605.05777 | 2026-05-07 | **57** | WEAK |

---

## Code Reproductions (Score ≥ 80)

| Paper | Model Name | Folder | Score |
|-------|-----------|--------|-------|
| ARGUS: Policy-Adaptive Ad Governance | `ARGUS` | `code/ARGUS/` | 84 |
| GLiGuard: Schema-Conditioned LLM Safeguard | `GLiGuard` | `code/GLiGuard/` | 80 |

---

## Paper Detail Links

| # | Paper File | arXiv |
|---|-----------|-------|
| 1 | [argus_ad_governance.md](papers/argus_ad_governance.md) | https://arxiv.org/abs/2605.02200 |
| 2 | [gliguard_llm_safeguard.md](papers/gliguard_llm_safeguard.md) | https://arxiv.org/abs/2605.07982 |
| 3 | [drip_r_retail_benchmark.md](papers/drip_r_retail_benchmark.md) | https://arxiv.org/abs/2605.07699 |
| 4 | [liscp_text_detection.md](papers/liscp_text_detection.md) | https://arxiv.org/abs/2605.05950 |
| 5 | [who_decides_harmful.md](papers/who_decides_harmful.md) | https://arxiv.org/abs/2605.01416 |
| 6 | [multimodal_data_curation.md](papers/multimodal_data_curation.md) | https://arxiv.org/abs/2605.01163 |
| 7 | [cost_of_context_mllm_rag.md](papers/cost_of_context_mllm_rag.md) | https://arxiv.org/abs/2605.05594 |
| 8 | [disaad_llm_uncertainty.md](papers/disaad_llm_uncertainty.md) | https://arxiv.org/abs/2605.05777 |
