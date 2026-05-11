# 电商内容生态 & 达人治理 Paper巡检 — 2026-05-08 (GMT+8)

> 巡检窗口: 2026-05-07T16:00Z ~ 2026-05-08T16:00Z (UTC) ≡ 2026-05-08 全天 (Asia/Shanghai)
> 数据来源: HuggingFace Daily Papers, arxiv cs.IR / cs.MM / cs.CL / cs.CV (2605.* prefix)
> 巡检时间: 2026-05-09 (UTC, 立即触发的首次运行)

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

## 候选池筛选

从 38 篇 HuggingFace Daily + cs.IR/cs.MM 列表中按以下规则筛选：
- **STRONG**: 直接涉及电商 / 内容理解 / 达人治理 / 违规检测 / 数据打标
- **WEAK**: 大厂高影响 LLM/VLM/MLLM 工作，可外推到电商场景
- 跳过安全/后门/对抗注入

## 入选论文 (10 篇) 与得分

| # | 标题 | arXiv | 得分 | 是否复现 | 是否进飞书卡片 |
|---|------|-------|------|---------|---------------|
| 1 | UniVA: Unified Value Alignment for Generative Recommendation | 2605.05803 | **86** | ✅ 已复现 | ✅ |
| 2 | CMTA: Cross-Modal Temporal Artifacts for AIGC Video Detection | 2605.00630 | **81** | ✅ 已复现 (官方仓库为空) | ✅ |
| 3 | TabEmbed: Generalist Embeddings for Tabular Understanding | 2605.04962 | 78 | — | ✅ |
| 4 | Multimodal Data Curation Through Ranked Retrieval | 2605.01163 | 77 | — | ✅ |
| 5 | Beyond Semantic Similarity: DCI for Agentic Search | 2605.05242 | 75 | — | ✅ |
| 6 | EKTM: Effective Knowledge Transfer for Multi-Task Recommendation | 2605.05730 | 71 | — | ✅ |
| 7 | GRE-MC: Robust Multimodal Recommendation via Graph Retrieval | 2605.00670 | 69 | — | ✅ |
| 8 | MiA-Signature: Approximating Global Activation for Long-Context | 2605.06416 | 69 | — | ✅ |
| 9 | IKEA.com Negative Data Mining for Dense Retrieval | 2605.00353 | 68 | — | ✅ |
| 10 | Crowdsourced Audiovisual Deepfake Detection | 2605.04797 | 44 | — | ✅ |

## 输出物

- `papers/<slug>.md` — 每篇论文的中文详细摘要 + 评分
- `feishu_cards.md` — 满足 ≥40 的飞书卡片格式汇总 (全部 10 篇)
- `code/UniVA/` — UniVA 的 PyTorch 玩具实现 (评分 86)
- `code/CMTA/` — CMTA 的 PyTorch 玩具实现 (评分 81，官方仓库为空仓库)
- `web/index.html` — 单页面 Web 应用，支持中英切换 / 评分滑块 / 日期选择 / 方法图与实验图开关
- `web/papers.json` — Web 应用的数据源
