# 2026-06-09 AI 论文巡检日报

**巡检时间**: 2026-06-10 08:30 GMT+8（覆盖 2026-06-09 GMT+8 全天索引的论文）

---

## Source Coverage 数据源覆盖情况

| source | category | attempted | http_status_or_error | candidates_yielded |
|--------|----------|-----------|----------------------|--------------------|
| arxiv cs.CL /list/new | aggregator | yes | 403 Forbidden | 0 (via WebFetch) |
| arxiv cs.CV /list/new | aggregator | yes | 403 Forbidden | 0 (via WebFetch) |
| arxiv cs.IR /list/new | aggregator | yes | 403 Forbidden | 0 (via WebFetch) |
| arxiv cs.MM /list/new | aggregator | yes | 403 Forbidden | 0 (via WebFetch) |
| arxiv cs.LG /list/new | aggregator | yes | 403 Forbidden | 0 (via WebFetch) |
| arxiv cs.AI /list/new | aggregator | yes | 403 Forbidden | 0 (via WebFetch) |
| arxiv cs.CL /pastweek | aggregator | yes | 403 Forbidden | 0 (via WebFetch) |
| HuggingFace papers/date/2026-06-09 | aggregator | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| HuggingFace papers/trending | aggregator | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| Google DeepMind blog | blog | yes | 403 Forbidden | 0 |
| Google Research blog | blog | yes | 403 Forbidden | 0 |
| Meta AI Research | blog | yes | 403 Forbidden | 0 |
| Anthropic News | blog | yes | 403 Forbidden | 0 |
| ByteDance Seed Research | blog | yes | 403 Forbidden | 0 |
| Meituan Tech | blog | yes | 403 Forbidden | 0 |
| Qwen Blog | blog | yes | 403 Forbidden | 0 (low prio) |
| 量子位 QbitAI | wechat | yes | 403 Forbidden | 0 (fallback WebSearch) |
| Semantic Scholar Graph API | api | yes | 403 Forbidden | 0 (Tier 2) |
| 机器之心 (WebSearch fallback) | wechat | yes | WebSearch ok, 0 in-scope | 0 |
| 新智元 (WebSearch fallback) | wechat | yes | WebSearch ok, 0 in-scope | 0 |
| OpenAI News (WebSearch fallback) | blog | yes | WebSearch ok | 0 (no new paper) |
| DeepSeek (WebSearch fallback) | blog | yes | WebSearch ok | 0 (no new paper) |
| Xiaohongshu/REDtech (WebSearch fallback) | blog | yes | WebSearch ok | 0 (no paper link) |
| Kuaishou Tech (WebSearch fallback) | blog | yes | WebSearch ok | 0 (no new paper Jun 9) |
| Tencent Hunyuan (WebSearch fallback) | blog | yes | WebSearch ok | 0 (no new paper Jun 9) |
| arxiv 2606 WebSearch (Tier 3 broad) | aggregator | yes | WebSearch ok | 8 new candidates |
| SIGIR 2026 accepted papers | conf | yes | 403 Forbidden (fallback WebSearch) | 2 candidates |
| SIGIR eCom 2026 | conf | yes | 403 Forbidden (fallback WebSearch) | 1 candidate |
| OpenReview ICLR/NeurIPS 2026 | conf | no | not yet active | 0 |

> **注**: 本次巡检所有 WebFetch 调用均返回 403 Forbidden（沙盒网络策略限制对外 HTTP 访问），全部来源均降级为 WebSearch 兜底。候选论文来自对 arxiv 2606 系列的系统性搜索。

---

## 本日精选（含评分）

新增论文（未在前次 2026-06-09 巡检中收录，补充自今日 2026-06-10 巡检）：

| 评分 | 论文 | arXiv | 关键词 |
|------|------|-------|--------|
| **88** | FLUID — 工业级直播推荐的多模态语义编码 | 2605.21832 | 直播推荐、多模态、语义 ID |
| **84** | CQ-SID+EG-GRPO — 电商搜索高效生成式召回 | 2605.14434 | 电商搜索、生成式召回、语义聚类 ID |
| **78** | QueryAgent-R1 — 基于 Chain-of-Retrieval 的电商查询推荐 | 2606.05671 | 电商、查询推荐、Agent |
| **76** | TGQ-Former — 文本引导视觉表征的多模态电商推荐 | 2605.17366 | 多模态推荐、视觉表征、电商 |
| **72** | Meta MLLM Framework — 工业级推荐系统的多模态理解通用框架 | 2605.09338 | MLLM、推荐系统、大规模 |
| **68** | SearchSwarm — 长程研究的 Agentic LLM 委派智能 | 2606.09730 | Agent、委派推理、搜索 |
| **62** | Unintended Consequences — 推荐系统干预的意外后果 | 2606.08265 | 推荐系统、A/B 实验、政策干预 |
| **55** | Memory Beyond Recall — LLM Agent 的双过程认知记忆 | 2606.09483 | Agent 记忆、双过程、Tencent |

> 前次巡检（2026-06-09 run）已收录：AdaGRPO(85)、MORE(83)、LCLM(78)、Cartridges(74)、RAG重写(71)、电商约束分配(70)、SAGE(67)、OmniCap-IF(68)、CoVEBench(64)、EvalCards(69)、SWE-Explore(62)、FlashMemory(69)。

---

## 评分机制（百分制）

总分 = 方法创新性(30) + 实验指标SOTA(15) + 实验质量/消融(15) + 方法效率(10) + 方法泛化性(5) + 论文相关性(25)。

---

## 代码复现说明（score >= 80）

- **FLUID**：原文使用内部工业数据，核心机制（LUCID 分层语义编码 + 无 ID 融合排序）已在 `code/FLUID/` 提供对齐论文公式的 toy 复现（数据/编码器/排序器/训练/评估完整）。
- **CQ-SID**：原文核心机制（CQ-SID 层次语义 ID + EG-GRPO 强化对齐）已在 `code/CQ-SID/` 提供 toy 复现。

---

## 快速导航

- [papers/fluid.md](papers/fluid.md) — FLUID 全文中文总结
- [papers/cq_sid.md](papers/cq_sid.md) — CQ-SID+EG-GRPO 全文中文总结
- [papers/queryagent_r1.md](papers/queryagent_r1.md) — QueryAgent-R1 全文中文总结
- [papers/tgq_former.md](papers/tgq_former.md) — TGQ-Former 全文中文总结
- [papers/meta_mllm_rec.md](papers/meta_mllm_rec.md) — Meta MLLM Framework 全文中文总结
- [papers/searchswarm.md](papers/searchswarm.md) — SearchSwarm 全文中文总结
- [papers/recommender_interventions.md](papers/recommender_interventions.md) — Recommender Interventions 全文中文总结
- [papers/memory_beyond_recall.md](papers/memory_beyond_recall.md) — Memory Beyond Recall 全文中文总结
- [feishu_cards.md](feishu_cards.md) — Feishu 推送卡片（score >= 40）
- [code/FLUID/](code/FLUID/) — FLUID 复现代码
- [code/CQ-SID/](code/CQ-SID/) — CQ-SID 复现代码
