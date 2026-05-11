# 📄 AI Paper Daily Inspection — 2026-05-10 (GMT+8) · MANUAL BACKFILL

**Domain:** E-commerce content ecosystem & influencer governance  
**Emphasis:** LLM / VLM / MLLM · captioning · vector embeddings · clustering & similarity · agents · RAG · distillation · violation detection · content governance · data quality

> **Backfill Note:** 2026-05-10 (GMT+8) is a **Sunday**. ArXiv does not announce new submissions on weekends; papers submitted Fri–Sun appear in Monday's listing. The sandbox network blocks most external HTTP/HTTPS endpoints (arXiv, HuggingFace, all lab blogs) with 403 Forbidden. All discovery was conducted via WebSearch fallback. Candidate window covers arXiv IDs in the 2605.01xxx–2605.07xxx range representing papers from the week of May 5–10, 2026. Papers with higher IDs (2605.08xxx+) correspond to post-Sunday announcements and were not yet indexed. This backfill does NOT duplicate the 2026-05-09 run (which already covered: 2605.07982, 2605.01416, 2605.05950, 2605.06285, 2605.07613, 2605.06619).

---

## Source Coverage Table

| Source | Category | Attempted | HTTP Status / Error | Candidates Yielded |
|--------|----------|-----------|--------------------|--------------------|
| arXiv cs.CL/pastweek | aggregator | yes | 403 Forbidden | 0 (WebSearch fallback: ~8) |
| arXiv cs.CV/pastweek | aggregator | yes | 403 Forbidden | 0 (WebSearch fallback: ~6) |
| arXiv cs.IR/pastweek | aggregator | yes | 403 Forbidden | 0 (WebSearch fallback: ~4) |
| arXiv cs.MM/pastweek | aggregator | yes | 403 Forbidden | 0 (WebSearch fallback: ~3) |
| arXiv cs.LG/pastweek | aggregator | yes | 403 Forbidden | 0 (WebSearch fallback: ~5) |
| arXiv cs.AI/pastweek | aggregator | yes | 403 Forbidden | 0 (WebSearch fallback: ~3) |
| HuggingFace Daily Papers 2026-05-10 | aggregator | yes | 403 Forbidden | 0 |
| HuggingFace Trending Papers | aggregator | yes | 403 Forbidden | 0 |
| Semantic Scholar Graph API | aggregator | yes | 403 Forbidden | 0 |
| Google DeepMind Blog | blog | yes | 403 Forbidden | 0 |
| Google Research Blog | blog | yes | 403 Forbidden | 0 |
| Meta AI Research | blog | yes | 403 Forbidden | 0 |
| Anthropic News | blog | yes | 403 Forbidden | 0 |
| ByteDance Seed Research | blog | yes | 403 Forbidden | 0 (Valley3 found via WebSearch) |
| Meituan Tech Blog | blog | yes | 403 Forbidden | 0 |
| Qwen Blog | blog | yes | 403 Forbidden (WebSearch: 200) | 0 (no new May 2026 posts found) |
| OpenAI News | blog | yes | 403 Forbidden (WebSearch: 200) | 1 (monitorability paper, out of scope) |
| DeepSeek | blog | yes | WebSearch: 200 | 1 (mHC paper, Jan 2026, out of window) |
| 量子位 QbitAI | wechat | yes | 403 Forbidden (WebSearch: 200) | 2 (Valley3 ref, AuDisAgent ref) |
| 机器之心 Jiqizhixin | wechat | yes | WebSearch: 200 | 1 (no direct May 10 paper refs) |
| 新智元 (WeChat) | wechat | yes | WebSearch: 200 | 0 (no links found) |
| 小红书 REDtech | blog | yes | WebSearch: 200 | 1 (EPIC embedding ref) |
| 快手 Kuaishou Tech | blog | yes | WebSearch: 200 | 0 (no new paper May 10) |
| 腾讯混元 Hunyuan | blog | yes | WebSearch: 200 | 0 (latest Hunyuan 3.0 from April 2026) |
| llm-stats.com AI News | aggregator | yes | 403 Forbidden | 0 |
| OpenReview ICLR/NeurIPS 2026 | conf | no | not yet active | 0 |
| WebSearch fallback (multi-topic) | aggregator | yes | 200 OK | **32 candidates total** |

**Summary:** 27 sources attempted; 0 direct HTTP success; WebSearch fallback yielded 32 candidates; 10 papers selected after dedup with 2026-05-09 run.

---

## Selected Papers — Scored

### STRONG (Directly E-com / Content Governance / Influencer)

| # | Title | arXiv ID | Score | Bucket |
|---|-------|----------|-------|--------|
| 1 | Valley3: Scaling Omni Foundation Models for E-commerce | 2605.01278 | **84** | STRONG ✦ |
| 2 | Omni-Fake: Benchmarking Unified Multimodal Social Media Deepfake Detection | 2605.01638 | **76** | STRONG |
| 3 | AuDisAgent: Multimodal Controversy Detection Multi-Agent Framework | 2605.02939 | **72** | STRONG |
| 4 | MultiSoc-4D: Instruction-Induced Label Collapse in LLM Annotation | 2605.06940 | **65** | STRONG |
| 5 | Multimodal Data Curation Through Ranked Retrieval (SNS+EEE) | 2605.01163 | **63** | STRONG |

### WEAK (High-Impact LLM/VLM/IR, Transferable to Ecom/Governance)

| # | Title | arXiv ID | Score | Bucket |
|---|-------|----------|-------|--------|
| 6 | The Cost of Context: Mitigating Textual Bias in Multimodal RAG (BAIR) | 2605.05594 | **66** | WEAK |
| 7 | SIRA: Superintelligent Retrieval Agent | 2605.06647 | **61** | WEAK |
| 8 | EPIC: Embedding-based In-Context Prompt Training | 2605.01372 | **60** | WEAK |
| 9 | Topic Is Not Agenda: Citation-Community Audit of Text Embeddings | 2605.07158 | **54** | WEAK |
| 10 | EGAD: Entropy-Guided Adaptive Distillation | 2605.01732 | **51** | WEAK |

---

## Code Reproduction (score ≥ 80)

| Paper | Score | Code Status |
|-------|-------|-------------|
| Valley3 (2605.01278) | 84 | `code/Valley3/` — full pipeline reproduced (toy data, omni MLLM architecture, 4-stage training) |

---

## Paper Detail Files

- [Valley3: Scaling Omni Foundation Models for E-commerce](papers/valley3_ecommerce_mllm.md)
- [Omni-Fake: Multimodal Social Media Deepfake Benchmark](papers/omni_fake_deepfake_benchmark.md)
- [AuDisAgent: Multimodal Controversy Detection](papers/audisagent_controversy_detection.md)
- [MultiSoc-4D: LLM Annotation Label Collapse](papers/multisoc4d_label_collapse.md)
- [Multimodal Data Curation Through Ranked Retrieval](papers/multimodal_data_curation_sns_eee.md)
- [The Cost of Context: Mitigating Textual Bias in Multimodal RAG](papers/cost_of_context_bair.md)
- [SIRA: Superintelligent Retrieval Agent](papers/sira_retrieval_agent.md)
- [EPIC: Embedding-based In-Context Prompt Training](papers/epic_embedding_ict.md)
- [Topic Is Not Agenda: Text Embeddings Audit](papers/topic_not_agenda_embeddings.md)
- [EGAD: Entropy-Guided Adaptive Distillation](papers/egad_distillation.md)
