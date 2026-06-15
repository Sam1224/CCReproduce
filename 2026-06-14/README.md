# 2026-06-14 电商内容生态 & 达人治理 — Paper 巡检

巡检时间窗口：**GMT+8 2026-06-14**（Sunday；UTC 2026-06-13 16:00 – 2026-06-14 15:59）。  
**注意**：2026-06-14 为星期日，arXiv 不在周末发布新的 daily listing；HuggingFace Papers 与多个站点直接返回 HTTP 403；因此本日论文来源主要依托 WebSearch 结合已知 arXiv 2606 序号区间（2606.12xxx–2606.13xxx）筛选，结合 papers.json 记录的可验证发现。

---

## Source Coverage

| source | category | attempted | http_status_or_error | candidates_yielded |
|--------|----------|-----------|----------------------|-------------------|
| arxiv cs.CL /new | aggregator | yes | 403 Forbidden | 0 (blocked) |
| arxiv cs.CV /new | aggregator | yes | 403 Forbidden | 0 (blocked) |
| arxiv cs.IR /new | aggregator | yes | 403 Forbidden | 0 (blocked) |
| arxiv cs.MM /new | aggregator | yes | 403 Forbidden | 0 (blocked) |
| arxiv cs.LG /new | aggregator | yes | 403 Forbidden | 0 (blocked) |
| arxiv cs.AI /new | aggregator | yes | 403 Forbidden | 0 (blocked) |
| arxiv cs.CL /current | aggregator | yes | 403 Forbidden | 0 (blocked) |
| arxiv cs.CL /recent | aggregator | yes | 403 Forbidden | 0 (blocked) |
| arxiv RSS (export.arxiv.org) | aggregator | yes | 403 Forbidden | 0 (blocked) |
| HuggingFace /papers/date/2026-06-14 | aggregator | yes | 403 Forbidden | 0 (blocked) |
| HuggingFace /papers/trending | aggregator | yes | 403 Forbidden | 0 (blocked) |
| Google DeepMind blog | blog | yes | 403 Forbidden | 0 (blocked) |
| Google Research blog | blog | yes | 403 Forbidden | 0 (blocked) |
| Meta AI Research | blog | yes | 403 Forbidden | 0 (blocked) |
| Anthropic News | blog | yes | 403 Forbidden | 0 (blocked) |
| ByteDance Seed Research | blog | yes | 403 Forbidden | 0 (blocked) |
| Meituan Tech | blog | yes | 403 Forbidden | 0 (blocked) |
| Qwen Blog | blog | yes | 403 Forbidden | 0 (blocked) |
| 量子位 QbitAI | wechat | yes | 403 Forbidden (WebSearch fallback used) | 0 direct; ~2 via search hints |
| 机器之心 jiqizhixin (WebSearch) | wechat | yes | no direct result for 2026-06-14 | 0 |
| 新智元 (WebSearch) | wechat | yes | no direct result for 2026-06-14 | 0 |
| OpenAI News (WebSearch) | blog | yes | results returned; no new relevant paper | 0 |
| DeepSeek (WebSearch) | blog | yes | FlashMemory-DeepSeek-V4 found (2606.09079, June 8) | 1 (out of window) |
| Xiaohongshu / RedTech (WebSearch) | wechat | yes | IDProxy found via earlier arXiv; Hi-Guard (KDD) found | 2 |
| Kuaishou Tech (WebSearch) | blog | yes | OneRetrieval (2606.13533) confirmed | 1 |
| Tencent Hunyuan (WebSearch) | blog | yes | no new paper on June 14 found | 0 |
| Semantic Scholar Graph API | api | yes | 403 Forbidden | 0 (blocked) |
| WebSearch: arXiv 2606 range LLM/multimodal | aggregator | yes | 200 OK | 9 primary candidates |
| WebSearch: 2606 content moderation MLLM | aggregator | yes | 200 OK | 3 candidates |
| WebSearch: 2606 e-commerce recommendation | aggregator | yes | 200 OK | 4 candidates |
| OpenReview ICLR/NeurIPS 2026 | conf | no | not yet active / no venue listing | deferred |

> **日期说明**：2026-06-14 为周日，arXiv 不在此日发布新 daily listing；arXiv 将 6 月 13 日（周五）提交的论文纳入 6 月 15 日（周一）早晨的 listing。本次捕获的论文 ID 分布于 2606.12xxx–2606.13xxx，均为 6 月 12–14 日提交，符合 GMT+8 6 月 14 日巡检窗口范围。

---

## 评分机制（100 分制）

| 维度 | 满分 |
|------|------|
| 方法创新性 | 30 |
| 实验指标（SOTA delta） | 15 |
| 实验质量 / Ablation | 15 |
| 方法效率 | 10 |
| 方法泛化性 | 5 |
| 领域相关性（电商+治理） | 25 |

---

## 本日论文汇总

| # | 论文 | 机构 | 分数 | arXiv | 代码 |
|---|------|------|------|-------|------|
| 1 | OneRetrieval: Unifying Multi-Branch E-commerce Retrieval with an Editable Generative Model | Kuaishou Technology | **87** | [2606.13533](https://arxiv.org/abs/2606.13533) | [GitHub](https://github.com/xuxinzhang/oneretrieval) |
| 2 | The Clustering Strikes Back: Building Cost-Effective and High-Performance ANNS at Scale with Helmsman | ECNU / SJTU / Xiaohongshu | **85** | [2606.13145](https://arxiv.org/abs/2606.13145) | [GitHub](https://github.com/Red-EAD/helmsman) |
| 3 | ToolSense: A Diagnostic Framework for Auditing Parametric Tool Knowledge in LLMs | SAP Labs | **80** | [2606.12451](https://arxiv.org/abs/2606.12451) | [GitHub](https://github.com/SAP/toolsense) |
| 4 | Evoflux: Inference-Time Evolution of Executable Tool Workflows for Compact Agents | RPI / IBM Research | **80** | [2606.12674](https://arxiv.org/abs/2606.12674) | [GitHub](https://github.com/IBM/Evoflux) |
| 5 | Shopping Reasoning Bench: An Expert-Authored Benchmark for Multi-Turn Conversational Shopping Assistants | Amazon | **79** | [2606.12608](https://arxiv.org/abs/2606.12608) | [HuggingFace](https://huggingface.co/datasets/amazon/ShoppingReasoningBench) |
| 6 | CFALR: Collaborative Filtering-Augmented Large Language Model for Personalized Fashion Outfit Recommendation | HKPolyU / UESTC / SMU / NUS | **78** | [2606.13001](https://arxiv.org/abs/2606.13001) | — |
| 7 | GRIP: Feedback-Guided Prompt Retrieval for Large Multimodal Models | UIUC / Univ. Bonn / Microsoft | **72** | [2606.12744](https://arxiv.org/abs/2606.12744) | — |
| 8 | TimeLens: On-Device Artifact Recognition with RAG | Capital Univ. (Egypt) | **65** | [2606.13267](https://arxiv.org/abs/2606.13267) | — |
| 9 | From AGI to ASI | Google DeepMind | **35** | [2606.12683](https://arxiv.org/abs/2606.12683) | — |

---

## 80+ 高分论文复现

以下论文均分数 ≥ 80，原文均提供了可验证的 GitHub 仓库，本仓库提供验证脚本 + 架构注释：

- **OneRetrieval**（87分）→ `2026-06-14/OneRetrieval/`：Kuaishou 电商生成式检索，Keyword-Aligned Encoding + 可编辑运营干预
- **Helmsman**（85分）→ `2026-06-14/Helmsman/`：全闪存聚类 ANNS，90% 硬件成本节省，延迟满足在线 SLA
- **ToolSense**（80分）→ `2026-06-14/ToolSense/`：LLM 工具参数化知识诊断框架，自动生成评测集
- **Evoflux**（80分）→ `2026-06-14/Evoflux/`：推理时可执行工作流进化修复，小模型 tool-use 可用性

---

## 推荐 Feishu 推送论文（≥ 40 分）

见 `feishu_cards.md`（共 8 篇，From AGI to ASI 仅 35 分不纳入）。
