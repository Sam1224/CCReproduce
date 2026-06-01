# 每日论文扫描 — 2026-05-31 (GMT+8)

> 域：电商内容生态 & 达人治理 | 重点：LLM/VLM/MLLM、内容理解、向量检索、Agent、RAG、蒸馏、违规检测、数据质量

---

## Source Coverage

| source | category | attempted | http_status_or_error | candidates_yielded |
|--------|----------|-----------|---------------------|-------------------|
| arxiv cs.CL/new | aggregator | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| arxiv cs.CV/new | aggregator | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| arxiv cs.IR/new | aggregator | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| arxiv cs.MM/new | aggregator | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| arxiv cs.LG/new | aggregator | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| arxiv cs.AI/new | aggregator | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| arxiv cs.CL/pastweek | aggregator | no | skipped (403 on /new) | 0 |
| arxiv cs.CV/pastweek | aggregator | no | skipped (403 on /new) | 0 |
| HuggingFace papers/date/2026-05-31 | aggregator | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| HuggingFace papers/trending | aggregator | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| Google DeepMind blog | blog | yes | 403 Forbidden | 0 |
| Google Research blog | blog | yes | 403 Forbidden | 0 |
| Meta AI Research | blog | yes | 403 Forbidden | 0 |
| Anthropic News | blog | yes | 403 Forbidden | 0 |
| ByteDance Seed Research | blog | yes | 403 Forbidden | 0 |
| Meituan Tech | blog | no | not attempted (403 expected) | 0 |
| Qwen Blog | blog | no | not attempted (403 expected) | 0 |
| 量子位 QbitAI | wechat | yes | 403 Forbidden | 0 (fallback to WebSearch) |
| 机器之心 (jiqizhixin.com) | wechat | yes (WebSearch fallback) | partial results | 1 |
| 新智元 | wechat | no (WebSearch only) | no recent results | 0 |
| OpenAI News | blog | no (WebSearch fallback) | no relevant results | 0 |
| DeepSeek | blog | no (WebSearch fallback) | no relevant results | 0 |
| Xiaohongshu/REDtech | wechat | no (WebSearch fallback) | no relevant results | 0 |
| Kuaishou Tech | blog | no (WebSearch fallback) | no relevant results | 0 |
| Tencent Hunyuan | blog | no (WebSearch fallback) | no relevant results | 0 |
| Semantic Scholar API | api | yes | 403 Forbidden | 0 |
| OpenReview ICLR/NeurIPS 2026 | conf | no | not yet active | 0 |
| **WebSearch fallback (arxiv 2605 domain)** | fallback | yes | 200 OK | 7 |

> ⚠ 环境网络策略屏蔽了绝大多数外部站点的直接 WebFetch，本次全部依赖 WebSearch 降级抓取。2026-05-31 为周日，arXiv 无新 submission 列表；候选论文来自本周（2026-05-25~31）的 2605.NNNNN 区段及近期发布的其他高相关论文。

---

## 候选论文总览

| # | arXiv ID | 标题 (简) | 桶 | 分数 |
|---|----------|----------|-----|------|
| 1 | [2512.03553](https://arxiv.org/abs/2512.03553) | Dynamic Content Moderation in Livestreams | STRONG | **85** |
| 2 | [2605.01278](https://arxiv.org/abs/2605.01278) | Valley3: Omni E-commerce MLLM | STRONG | **81** |
| 3 | [2605.17366](https://arxiv.org/abs/2605.17366) | TGQ-Former: Text-Guided E-commerce Recommendation | STRONG | **75** |
| 4 | [2601.16027](https://arxiv.org/abs/2601.16027) | Deja Vu in Plots: RAG for Livestream Risk | STRONG | **76** |
| 5 | [2605.07760](https://arxiv.org/abs/2605.07760) | RuleSafe-VL: Rule-Conditioned Content Moderation | STRONG | **71** |
| 6 | [2605.09338](https://arxiv.org/abs/2605.09338) | MLLM Framework for Large-Scale Recommendation | STRONG | **69** |
| 7 | [2605.07982](https://arxiv.org/abs/2605.07982) | GLiGuard: Schema-Conditioned LLM Safeguard | WEAK→STRONG | **69** |

---

## 详细论文索引

- [papers/livestream_mod.md](papers/livestream_mod.md) — Dynamic Content Moderation in Livestreams (Score: 85) 🔥
- [papers/valley3.md](papers/valley3.md) — Valley3 Omni E-commerce MLLM (Score: 81) 🔥
- [papers/tgqformer.md](papers/tgqformer.md) — TGQ-Former E-commerce Recommendation (Score: 75)
- [papers/deja_vu_plots.md](papers/deja_vu_plots.md) — Deja Vu in Plots RAG Livestream (Score: 76)
- [papers/rulesafe_vl.md](papers/rulesafe_vl.md) — RuleSafe-VL Content Moderation (Score: 71)
- [papers/mllm_recsys.md](papers/mllm_recsys.md) — MLLM Framework for Recommendation (Score: 69)
- [papers/gliguard.md](papers/gliguard.md) — GLiGuard LLM Safeguard (Score: 69)

---

## 代码复现（Score ≥ 80）

- [code/LivestreamMod/](code/LivestreamMod/) — Dynamic Content Moderation in Livestreams 复现
- [code/Valley3/](code/Valley3/) — Valley3 Omni E-commerce MLLM 复现（toy 级）
