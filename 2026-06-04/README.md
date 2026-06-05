# 电商内容生态 & 达人治理 Paper巡检 — 2026-06-04 (GMT+8)

> 巡检窗口: 2026-06-03T16:00Z ~ 2026-06-04T16:00Z (UTC) ≡ 2026-06-04 全天 (Asia/Shanghai)
> 数据来源: arXiv cs.CL / cs.CV / cs.IR / cs.MM / cs.LG / cs.AI (2606.04XXX prefix 为主窗口)
> 巡检时间: 2026-06-05 00:30 CST (scheduled)

---

## Source Coverage

| source | category | attempted | http_status_or_error | candidates_yielded |
|--------|----------|-----------|----------------------|---------------------|
| arxiv cs.CL/new | aggregator | yes | 403 Forbidden | 0 (WebSearch fallback used) |
| arxiv cs.CV/new | aggregator | yes | 403 Forbidden | 0 (WebSearch fallback used) |
| arxiv cs.IR/new | aggregator | yes | 403 Forbidden | 0 (WebSearch fallback used) |
| arxiv cs.MM/new | aggregator | yes | 403 Forbidden | 0 (WebSearch fallback used) |
| arxiv cs.LG/new | aggregator | yes | 403 Forbidden | 0 (WebSearch fallback used) |
| arxiv cs.AI/new | aggregator | yes | 403 Forbidden | 0 (WebSearch fallback used) |
| arxiv cs.AI/pastweek | aggregator | yes | 403 Forbidden | 0 (WebSearch fallback used) |
| huggingface.co/papers/date/2026-06-04 | aggregator | yes | 403 Forbidden | 0 |
| huggingface.co/papers/trending | aggregator | yes | 403 Forbidden | 0 |
| deepmind.google/discover/blog | blog | yes | 403 Forbidden | 0 |
| research.google/blog | blog | yes | 403 Forbidden | 0 |
| ai.meta.com/research | blog | yes | 403 Forbidden | 0 |
| anthropic.com/news | blog | yes | 403 Forbidden | 0 |
| seed.bytedance.com/research | blog | yes | 403 Forbidden | 0 |
| tech.meituan.com | blog | yes | 403 Forbidden | 0 |
| qwenlm.github.io/blog | blog | yes | 403 Forbidden | 0 |
| qbitai.com (量子位) | wechat | yes | 403 Forbidden | 0 (WebSearch fallback used) |
| 机器之心 jiqizhixin.com | wechat | no (WebSearch) | no recent 2026-06 content indexed | 0 |
| 新智元 mp.weixin.qq.com | wechat | no (WebSearch) | partial; no specific paper refs | 0 |
| openai.com/news | blog | no (WebSearch) | gpt-oss-120b release found, not a research paper | 0 |
| DeepSeek site:deepseek.com | blog | no (WebSearch) | mHC paper found from Jan 2026 only | 0 |
| 小红书/REDtech | wechat | no (WebSearch) | no relevant papers found | 0 |
| Kuaishou Tech | wechat | no (WebSearch) | LiveForesighter / unbiased reranking found (outside date window) | 1 (out-of-window) |
| Tencent Hunyuan | wechat | no (WebSearch) | no June 2026 specific paper found | 0 |
| Semantic Scholar Graph API | api | yes | 403 Forbidden | 0 |
| WebSearch (arxiv 2606.04XXX discovery) | fallback | yes | 200 | 7 |
| OpenReview ICLR/NeurIPS 2026 | conf | no | not yet active | 0 |

> **注**: arxiv.org 对本次运行的所有 WebFetch 调用返回 HTTP 403 Forbidden (包括 /list、/abs、/html、/pdf 路径)。已切换为 WebSearch 作为全量 fallback，通过搜索结果 snippet 提取候选论文信息。因此对各候选论文的完整 abstract、详细指标、submission 准确日期只能依赖搜索引擎 snippet；不可获取时已标注"待论文细看"。

---

## 评分体系

| 维度 | 满分 |
|------|------|
| 方法创新性 | 30 |
| 实验指标 (SOTA delta) | 15 |
| 实验质量 (ablation) | 15 |
| 方法效率 | 10 |
| 方法泛化性 | 5 |
| 论文相关性 (电商+治理) | 25 |
| **合计** | **100** |

---

## 候选筛选

从 WebSearch 在 arxiv 2606.04XXX 范围内命中约 20 篇，聚焦 LLM agent / governance / recommendation / personalization 关键词，按以下规则分桶：

- **STRONG**: 直接涉及电商 / 内容治理 / 达人生态 / 违规检测 / 数据打标 / live commerce
- **WEAK**: 大厂高影响 LLM/VLM/MLLM 工作，可外推到电商或治理场景
- 跳过 offensive ML (backdoor/poisoning/membership-inference)

---

## 入选论文 (7 篇) 与得分

| # | 标题 | arXiv | 桶位 | 得分 | 是否复现 | 飞书卡片 |
|---|------|-------|------|------|---------|---------|
| 1 | OCL: Organizational Control Layer — Governance Infrastructure at the Execution Boundary of LLM Agent Systems | 2606.04306 | STRONG | **80** | ✅ `code/OCL/` | ✅ |
| 2 | E-VAds: An E-commerce Short Videos Understanding Benchmark for MLLMs | 2602.08355 | STRONG | 78 | — | ✅ |
| 3 | TAP-PER: Beyond Retrieval — Compact User Representations for Scalable LLM Personalization | 2606.04547 | STRONG | 70 | — | ✅ |
| 4 | LARM: LLM-Alignment Live-Streaming Recommendation | 2504.05217 | STRONG | 72 | — | ✅ |
| 5 | RUBAS: Rubric-Based Reinforcement Learning for Agent Safety | 2606.04051 | WEAK | 62 | — | ✅ |
| 6 | PHF: Beyond Isolated Behaviors — Hierarchical User Modeling for LLM Personalization | 2606.02300 | WEAK | 62 | — | ✅ |
| 7 | Can LLMs Clean Up Your Mess? Survey of Application-Ready Data Preparation | 2601.17058 | WEAK | 58 | — | ✅ |

> **日期说明**: 因 arXiv 列表页全部返回 403，无法直接确认各论文的 per-paper submission timestamp。以 arXiv ID 前缀 2606.04XXX 系列作为"2026-06-04 (GMT+8) 公告"的近似。E-VAds (2602.08355) 和 LARM (2504.05217) 源自发现阶段的 supplementary catch-up，真实提交日期分别为 2026-02 和 2026-04，已在对应 paper 文件中注明。

---

## 输出物

- `papers/ocl.md` — OCL 中文详细摘要 + 评分
- `papers/evads.md` — E-VAds 中文详细摘要 + 评分
- `papers/tap_per.md` — TAP-PER 中文详细摘要 + 评分
- `papers/larm.md` — LARM 中文详细摘要 + 评分
- `papers/rubas.md` — RUBAS 中文详细摘要 + 评分
- `papers/phf.md` — PHF 中文详细摘要 + 评分
- `papers/llms_data_prep.md` — LLMs4DataPrep Survey 中文详细摘要 + 评分
- `feishu_cards.md` — 全部 7 篇飞书卡片（评分 ≥40）
- `code/OCL/` — OCL 的 PyTorch 玩具复现（评分 80）
- `../web/index.html` — 单页面 Web 应用（中英切换 / 评分滑块 / 日期选择）
