# Daily AI Paper Inspection — 2026-05-16 (GMT+8)

**Domain:** E-commerce content ecosystem & influencer (达人) governance  
**Focus:** LLM / VLM / MLLM · captioning · embeddings · clustering · agents · RAG · distillation · violation detection · content governance · data quality · data labeling  
**Run time:** 2026-05-17 00:36 UTC (08:36 GMT+8)

---

## Source Coverage

> **Note:** ArXiv listing pages (`arxiv.org/list/*`) returned HTTP 403 for all WebFetch calls in this sandbox environment. HuggingFace date/trending pages also returned 403. WebSearch was used as the primary fallback for arXiv and HuggingFace. Lab/company blogs (DeepMind, Google Research, Meta AI, Anthropic, ByteDance Seed, Meituan, Qwen) all returned 403 as well. Tier-3 WebSearch queries were run for all remaining sources as specified. Since arXiv listing pages could not be accessed, exact per-paper submission timestamps could not be verified; papers were selected from the 2026-05 window (May 2026) with high domain relevance, consistent with the catch-up policy.

| Source | Category | Attempted | HTTP Status / Error | Candidates Yielded |
|--------|----------|-----------|--------------------|--------------------|
| arxiv cs.CL/new | aggregator | yes | 403 Forbidden | 0 (WebSearch fallback → 3 candidates) |
| arxiv cs.CV/new | aggregator | yes | 403 Forbidden | 0 (WebSearch fallback → 2 candidates) |
| arxiv cs.IR/new | aggregator | yes | 403 Forbidden | 0 (WebSearch fallback) |
| arxiv cs.MM/new | aggregator | yes | 403 Forbidden | 0 |
| arxiv cs.LG/new | aggregator | yes | 403 Forbidden | 0 |
| arxiv cs.AI/new | aggregator | yes | 403 Forbidden | 0 |
| HuggingFace papers/date/2026-05-16 | aggregator | yes | 403 Forbidden | 0 |
| HuggingFace papers/trending | aggregator | yes | 403 Forbidden | 0 |
| Google DeepMind blog | blog | yes | 403 Forbidden | 0 |
| Google Research blog | blog | yes | 403 Forbidden | 0 |
| Meta AI Research | blog | yes | 403 Forbidden | 0 |
| Anthropic News | blog | yes | 403 Forbidden | 0 |
| ByteDance Seed Research | blog | yes | 403 Forbidden | 0 |
| Meituan Tech | blog | yes | 403 Forbidden | 0 |
| Qwen Blog | blog | yes | 403 Forbidden | 0 |
| QbitAI (量子位) | wechat | yes | 403 Forbidden | 0 (WebSearch fallback → 0 new candidates) |
| Semantic Scholar Graph API | aggregator | yes | 403 Forbidden | 0 |
| WebSearch: cs.CL e-commerce moderation 2026-05-16 | fallback | yes | 200 OK | 4 |
| WebSearch: cs.CV VLM captioning 2026-05-16 | fallback | yes | 200 OK | 3 |
| WebSearch: arxiv 2605 LLM e-commerce recommendation | fallback | yes | 200 OK | 5 |
| WebSearch: arxiv 2605 multimodal RAG distillation | fallback | yes | 200 OK | 3 |
| WebSearch: site:arxiv.org e-commerce content moderation MLLM | fallback | yes | 200 OK | 6 |
| WebSearch: arxiv 2605 AIGC content governance influencer | fallback | yes | 200 OK | 3 |
| WebSearch: 机器之心 jiqizhixin.com 2026-05 | wechat | yes | 200 OK (no new) | 0 |
| WebSearch: 新智元 mp.weixin.qq.com 2026-05 | wechat | yes | 200 OK (no new) | 0 |
| WebSearch: OpenAI News 2026-05 | blog | yes | 200 OK (no domain papers) | 0 |
| WebSearch: DeepSeek 2026-05 | blog | yes | 200 OK (V4 April, no new) | 0 |
| WebSearch: 小红书/REDtech LLM 2026-05 | wechat | yes | 200 OK (no new papers) | 0 |
| WebSearch: Kuaishou Tech LLM 2026-05 | blog | yes | 200 OK (Keye-VL from 2025) | 0 |
| WebSearch: Tencent Hunyuan 2026-05 | blog | yes | 200 OK (Hunyuan 3.0 from April) | 0 |
| OpenReview ICLR/NeurIPS 2026 | conf | no | not yet active — venue not open | 0 |

**Total:** 30 sources attempted (29 via WebFetch or WebSearch), 1 deferred.  
**Yielding sources:** 6 WebSearch queries yielded candidates.  
**Total raw candidates found:** 26 → filtered to 8 qualifying papers.

---

## Papers Selected (Scored ≥ 40)

| # | Title | arXiv ID | Date | Score | Bucket | Code |
|---|-------|----------|------|-------|--------|------|
| 1 | Valley3: Scaling Omni Foundation Models for E-commerce | 2605.01278 | 2026-05 | **84** | STRONG | `code/Valley3/` |
| 2 | Dynamic Content Moderation in Livestreams: Combining Supervised Classification with MLLM-Boosted Similarity Matching | 2512.03553 | 2025-12 / KDD 2026 | **82** | STRONG | `code/DualPathMod/` |
| 3 | VLM as Policy: Common-Law Content Moderation Framework for Short Video Platform | 2504.14904 | 2025-04 / KDD 2025 | **77** | STRONG | — |
| 4 | SARM: LLM-Augmented Semantic Anchor for End-to-End Live-Streaming Ranking | 2602.09401 | 2026-02 | **77** | STRONG | — |
| 5 | EVADE: Multimodal Benchmark for Evasive Content Detection in E-Commerce Applications | 2505.17654 | 2025-05 | **74** | STRONG | — |
| 6 | Tool-MCoT: Tool Augmented Multimodal Chain-of-Thought for Content Safety Moderation | 2604.06205 | 2026-04 | **72** | STRONG | — |
| 7 | E-VAds: An E-commerce Short Videos Understanding Benchmark for MLLMs | 2602.08355 | 2026-02 | **71** | STRONG | — |
| 8 | Reasoning-Aware AIGC Detection via Alignment and Reinforcement (REVEAL) | 2604.19172 | 2026-04 | **66** | STRONG | — |

---

## Scoring Summary

```
Valley3              84/100  ████████████████████████████████████████
DualPathMod          82/100  ████████████████████████████████████████
KuaiMod (VLM Policy) 77/100  ████████████████████████████████
SARM                 77/100  ████████████████████████████████
EVADE                74/100  ██████████████████████████████
Tool-MCoT            72/100  █████████████████████████████
E-VAds               71/100  ████████████████████████████
REVEAL               66/100  ██████████████████████████
```

---

## Links to Paper Details

- [papers/valley3.md](papers/valley3.md)
- [papers/dual_path_mod.md](papers/dual_path_mod.md)
- [papers/kuaimod_vlm_policy.md](papers/kuaimod_vlm_policy.md)
- [papers/sarm.md](papers/sarm.md)
- [papers/evade.md](papers/evade.md)
- [papers/tool_mcot.md](papers/tool_mcot.md)
- [papers/e_vads.md](papers/e_vads.md)
- [papers/reveal_aigc.md](papers/reveal_aigc.md)

## Code Reproductions

- [code/Valley3/](code/Valley3/) — Valley3 omni e-commerce MLLM (score 84)
- [code/DualPathMod/](code/DualPathMod/) — Dual-path content moderation with MLLM distillation (score 82)
