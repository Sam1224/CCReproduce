# Daily AI Paper Inspection — 2026-05-20 (GMT+8)

> **Run time:** 2026-05-21 00:30 UTC (08:30 GMT+8)  
> **Domain focus:** E-commerce content ecosystem & influencer (达人) governance  
> **Key areas:** LLM / VLM / MLLM, captioning, embeddings, clustering, agents, RAG, distillation, violation detection, content governance, data quality, data labeling

---

## Source Coverage

> **Note on arxiv access:** All arxiv.org WebFetch calls returned HTTP 403 (blocked in sandbox). Discovery relied on WebSearch fallback for arxiv, which surfaces indexed papers from throughout May 2026 (2605.xxxxx range). Time-window filtering to precisely 2026-05-20 was best-effort; papers are from throughout May 2026 unless a confirmed submission date is noted.

| source | category | attempted | http_status_or_error | candidates_yielded |
|--------|----------|-----------|----------------------|--------------------|
| arxiv cs.CL/new | aggregator | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| arxiv cs.CV/new | aggregator | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| arxiv cs.IR/new | aggregator | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| arxiv cs.MM/new | aggregator | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| arxiv cs.LG/new | aggregator | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| arxiv cs.AI/new | aggregator | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| arxiv WebSearch fallback (cs.CL/CV/IR/LG/AI) | aggregator | yes | search_ok | 9 candidates from 2605.* |
| HuggingFace papers/date/2026-05-20 | aggregator | yes | 403 Forbidden | 0 |
| HuggingFace papers/trending | aggregator | yes | 403 Forbidden | 0 |
| HuggingFace WebSearch fallback | aggregator | yes | search_ok | 0 relevant |
| Google DeepMind blog | blog | yes | 403 Forbidden | 0 |
| Google Research blog | blog | yes | 403 Forbidden | 0 |
| Meta AI Research | blog | yes | 403 Forbidden | 0 |
| Anthropic News | blog | yes | 403 Forbidden | 0 |
| ByteDance Seed Research | blog | yes | 403 Forbidden | 0 |
| Meituan Tech | blog | yes | 403 Forbidden | 0 |
| Qwen Blog | blog | yes | 403 Forbidden (low priority) | 0 |
| 量子位 QbitAI (qbitai.com) | wechat | yes | 403 Forbidden | 0 |
| 机器之心 (jiqizhixin.com) | wechat | yes (WebSearch) | search_ok | 0 new relevant |
| 新智元 (mp.weixin.qq.com) | wechat | yes (WebSearch) | search_ok, JS-rendered | 0 |
| OpenAI News | blog | yes (WebSearch) | search_ok | 0 domain-relevant |
| DeepSeek | blog | yes (WebSearch) | search_ok | 0 domain-relevant |
| Xiaohongshu REDtech | wechat | yes (WebSearch) | search_ok | 0 new papers |
| Kuaishou Tech | blog | yes (WebSearch) | search_ok | 0 new papers |
| Tencent Hunyuan | blog | yes (WebSearch) | search_ok | 0 new papers |
| Semantic Scholar Graph API | aggregator | yes | 403 Forbidden | 0 |
| OpenReview ICLR/NeurIPS 2026 | conf | no | not yet active | 0 |

---

## Paper Summary (May 2026)

Papers are from the `2605.xxxxx` arXiv namespace (May 2026). Confirmed submission dates noted where available.

| # | Title | arXiv ID | Bucket | Score | Confirmed Date |
|---|-------|----------|--------|-------|----------------|
| 1 | RuleSafe-VL: Evaluating Rule-Conditioned Decision Reasoning in VL Content Moderation | 2605.07760 | STRONG | **81** | May 8, 2026 |
| 2 | GLiGuard: Schema-Conditioned Classification for LLM Safeguard | 2605.07982 | STRONG | **81** | May 13, 2026 |
| 3 | CMTA: Cross-Modal Temporal Artifacts for Generalizable AI-Generated Video Detection | 2605.00630 | STRONG | **77** | May 1, 2026 |
| 4 | LiSCP: Lightweight Stylistic Consistency Profiling for LLM-Generated Text Detection | 2605.05950 | STRONG | **75** | ~May 7, 2026 |
| 5 | GRE-MC: Robust Multimodal Recommendation via Graph Retrieval-Enhanced Modality Completion | 2605.00670 | STRONG | **70** | May 1, 2026 |
| 6 | PluRule: A Benchmark for Moderating Pluralistic Communities on Social Media | 2605.17187 | STRONG | **70** | May 16, 2026 |
| 7 | Who Decides What Is Harmful? Multi-Agent Personalised Inference for Content Moderation | 2605.01416 | STRONG | **68** | May 2, 2026 |
| 8 | VLMs as Weak Annotators in Active Learning | 2605.00480 | WEAK | **62** | May 1, 2026 |
| 9 | SkillsVote: Lifecycle Governance of Agent Skills | 2605.18401 | WEAK | **58** | May 18, 2026 |
| 10 | Chain-based Distillation (CBD) for Variable-Sized SLMs | 2605.07783 | WEAK | **55** | May 8, 2026 |

---

## Code Reproduction

Papers with score ≥ 80 require code reproduction:
- **RuleSafe-VL** (81) → `code/RuleSafe-VL/`
- **GLiGuard** (81) → Real code available at [github.com/fastino-ai/GLiGuard](https://github.com/fastino-ai/GLiGuard) (model weights via GLiNER2); `code/GLiGuard/` contains usage examples and a training framework sketch.

---

## Notes

- All Tier 1 WebFetch sources returned HTTP 403 (Forbidden). This appears to be a sandbox-level network restriction where external content-delivery sites block non-browser requests.
- Discovery pivoted to WebSearch-based fallback across all sources.
- The arXiv `2605.` prefix corresponds to May 2026. Papers from the end of May 20 would have IDs in the `2605.19xxx–2605.20xxx` range; most found papers have earlier IDs (May 1–18) as web indexing of very recent submissions lags by 1–2 days.
- No EVADE paper (2505.17654) was included — its ID indicates May 2025, outside the window.
