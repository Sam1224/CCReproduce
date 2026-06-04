# 2026-06-03 Daily AI Paper Inspection

**Domain Focus:** E-commerce content ecosystem & influencer (达人) governance  
**Date (GMT+8):** 2026-06-03 (Yesterday)  
**Run time:** 2026-06-04 08:33 GMT+8  
**Emphasis:** LLM/VLM/MLLM · Captioning · Vector Embeddings · Clustering & Similarity · Agents · RAG · Distillation · Violation Detection · Content Governance · Data Quality · Data Labeling

---

## Source Coverage

| source | category | attempted | http_status_or_error | candidates_yielded |
|--------|----------|-----------|---------------------|-------------------|
| arxiv cs.CL/new | aggregator | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| arxiv cs.CV/new | aggregator | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| arxiv cs.IR/new | aggregator | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| arxiv cs.MM/new | aggregator | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| arxiv cs.LG/new | aggregator | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| arxiv cs.AI/current | aggregator | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| arxiv cs/recent | aggregator | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| HuggingFace papers/2026-06-03 | aggregator | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| HuggingFace trending | aggregator | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| Google DeepMind blog | blog | yes | 403 Forbidden | 0 |
| Google Research blog | blog | yes | 403 Forbidden | 0 |
| Meta AI Research | blog | yes | 403 Forbidden | 0 |
| Anthropic News | blog | yes | 403 Forbidden | 0 |
| ByteDance Seed Research | blog | yes | 403 Forbidden | 0 |
| Meituan Tech | blog | no | not attempted (URL unavailable in sandbox) | 0 |
| Qwen Blog | blog | no | not attempted (known 403 pattern) | 0 |
| 量子位 QbitAI | wechat | yes (WebSearch fallback) | accessible via WebSearch | 1 (M3-Agent, arxiv:2508.09736, out of scope date) |
| 机器之心 Jiqizhixin | wechat | yes (WebSearch fallback) | accessible via WebSearch | 0 new June-2026 papers |
| 新智元 Xinzhiyuan | wechat | yes (WebSearch fallback) | accessible via WebSearch | 0 new June-2026 papers |
| OpenAI News | blog | yes (WebSearch fallback) | accessible via WebSearch | 0 new relevant papers |
| DeepSeek | blog | yes (WebSearch fallback) | accessible via WebSearch | 0 new papers |
| Xiaohongshu / RedTech | blog | yes (WebSearch fallback) | accessible via WebSearch | 0 |
| Kuaishou Tech | blog | yes (WebSearch fallback) | accessible via WebSearch | 0 |
| Tencent Hunyuan | blog | yes (WebSearch fallback) | accessible via WebSearch | 0 |
| Semantic Scholar Graph API | aggregator | yes | 403 Forbidden | 0 |
| arxiv cs.CL/pastweek (catch-up) | aggregator | no | skipped (403 pattern established) | — |
| WebSearch arXiv 2606.xxxxx multi-query | aggregator | yes | 200 OK (search) | 14 candidates |
| OpenReview ICLR/NeurIPS 2026 | conf | no | not yet active | — |

**Discovery method note:** All direct arxiv.org WebFetch attempts returned HTTP 403. All direct lab blog/aggregator WebFetch attempts returned HTTP 403. Full discovery was accomplished via **WebSearch** (Tier 3 fallback), iterating over domain keywords × June 2026 date constraints. All 14 candidates found have arxiv IDs in the `2606.xxxxx` series (June 2026) or are recent high-relevance papers (SIGIR 2026 paper `2605.09338`).

---

## Picked Papers

| # | Paper | arXiv | Bucket | Score | Code |
|---|-------|-------|--------|-------|------|
| 1 | PaSBench-Video: A Streaming Video Benchmark for Proactive Safety Warning | [2606.02443](https://arxiv.org/abs/2606.02443) | STRONG | **85** | `code/PaSBench/` |
| 2 | A General Framework for Multimodal LLM-Based Multimedia Understanding in Large-Scale Recommendation Systems | [2605.09338](https://arxiv.org/abs/2605.09338) | STRONG | **82** | `code/MultimodalRecFramework/` |
| 3 | Investigating and Alleviating Harm Amplification in LLM Interactions | [2606.02423](https://arxiv.org/abs/2606.02423) | STRONG | **81** | `code/TrajSafe/` |
| 4 | RCEM: Embedder Equipped with Query Rewriting Skill for Robust Conversational Search in Distributional Shift | [2606.01697](https://arxiv.org/abs/2606.01697) | STRONG | **75** | — |
| 5 | DataShield: Safety-degrading Data Filtering for LLM Benign Instruction Fine-Tuning | [2606.00160](https://arxiv.org/abs/2606.00160) | STRONG | **74** | — (official code: https://github.com/ZJunBo/DataShield) |
| 6 | Beyond Isolated Behaviors: Hierarchical User Modeling for LLM Personalization | [2606.02300](https://arxiv.org/abs/2606.02300) | STRONG | **70** | — |
| 7 | Mitigating Perceptual Judgment Bias in Multimodal LLM-as-a-Judge via Perceptual Perturbation and Reward Modeling | [2606.02578](https://arxiv.org/abs/2606.02578) | WEAK | **68** | — |
| 8 | SkillAdaptor: Self-Adapting Skills for LLM Agents from Trajectories | [2606.01311](https://arxiv.org/abs/2606.01311) | WEAK | **62** | — |
| 9 | Moment-Video: Diagnosing Temporal Fidelity of Video MLLMs on Momentary Visual Events | [2606.02522](https://arxiv.org/abs/2606.02522) | WEAK | **61** | — |

---

## Papers Detail Index

- [PaSBench-Video](papers/pasbench_video.md)
- [MultimodalRecFramework](papers/multimodal_rec_framework.md)
- [HarmAmp_TrajSafe](papers/harmamp_trajsafe.md)
- [RCEM](papers/rcem.md)
- [DataShield](papers/datashield.md)
- [PHF_UserModeling](papers/phf_user_modeling.md)
- [MLLM_JudgeBias](papers/mllm_judge_bias.md)
- [SkillAdaptor](papers/skilladaptor.md)
- [Moment_Video](papers/moment_video.md)
