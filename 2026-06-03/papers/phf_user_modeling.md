# Beyond Isolated Behaviors: Hierarchical User Modeling for LLM Personalization

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | Beyond Isolated Behaviors: Hierarchical User Modeling for LLM Personalization |
| **Authors** | Liang Wang, Xinyi Mou, Xiaoyou Liu, Tiannan Wang, Yuqing Wang, Zhongyu Wei |
| **Affiliation** | Fudan University; Shanghai Innovation Institute; OPPO |
| **arXiv** | https://arxiv.org/abs/2606.02300 |
| **Submitted** | 2026-06-01 (appears in June 3, 2026 GMT+8 listing) |
| **Domain** | User Modeling · LLM Personalization · Recommendation · Behavioral Analysis |
| **Code** | — |

---

## 方法概述 / Method Overview

### 问题 / Problem
现有 LLM 个性化方法大多采用**扁平行为范式（flat behavioral paradigm）**：直接聚合用户历史行为，而未考虑这些行为如何组织成更深层的行为结构。这导致个性化系统忽视了用户行为的时序积累、跨用户的共性规律，以及行为背后的稳定偏好。

Existing LLM personalization methods use a **flat behavioral paradigm**: directly aggregating user behavior without modeling how behaviors organize into deeper structures — missing temporal accumulation, cross-user patterns, and stable dispositions.

### 方法 / Method

**PHF（Practice-Habitus-Field）框架**，基于社会学家皮埃尔·布迪厄的实践论（Theory of Practice），将 LLM 个性化重新概念化为三个层次：

| 层次 | 社会学概念 | 技术含义 |
|------|-----------|---------|
| Practice（实践） | 个体具体行为 | 用户的单次交互记录 |
| Habitus（惯习） | 行为随时间累积形成的稳定倾向 | 用户的长期偏好模式（用户级表示） |
| Field（场域） | 相似用户群体的共性规律 | 跨用户的聚类共享知识（群体级表示） |

**推理阶段：** 根据聚类质心将用户分配到最近的 Field，无需访问其他用户数据，保护隐私。

**Story Arc:** "LLM 个性化被扁平行为表示所限 → PHF 借助社会学层次结构，从实践到惯习到场域，实现更深层的用户理解"

*LLM personalization is limited by flat behavior representation → PHF uses sociological hierarchy (Practice → Habitus → Field) for deeper user understanding.*

---

## 创新性分析 / Innovation

1. **跨学科创新**：将社会学布迪厄实践论引入 LLM 个性化，提供了有理论根基的用户建模框架，而非单纯的工程启发式方法。
2. **三层次建模**：明确区分"单次行为"、"个人倾向"和"群体规律"三个层次，比现有扁平聚合更具表达力。
3. **隐私友好推理**：推理时仅使用聚类质心，无需实时访问其他用户数据。
4. **与电商/社交推荐的自然对应**：Practice = 浏览/购买记录，Habitus = 消费偏好，Field = 用户群体标签（达人粉丝、买家群体等）。

---

## 关键指标 / Key Metrics

| Task | Metric | PHF | Flat Baseline |
|------|--------|-----|---------------|
| Personalized Text Generation | ROUGE/BERTScore | Reported improvement | Prior personalization |
| User Preference Alignment | User Rating | Reported improvement | — |

*Exact numbers not retrieved due to arXiv 403.*

---

## 评分 / Scoring

| Dimension | Max | Score | Justification |
|-----------|-----|-------|---------------|
| Innovation | 30 | 20 | Novel sociological grounding; three-level hierarchy with clear semantic interpretability |
| SOTA Delta | 15 | 10 | Personalization improvements on text generation tasks |
| Exp. Quality | 15 | 10 | Multi-task personalization evaluation |
| Efficiency | 10 | 6 | Clustering-based field assignment at inference; no extra LLM calls per user |
| Generalization | 5 | 4 | Applicable across multiple personalization tasks |
| Domain Relevance | 25 | 20 | Directly relevant to user modeling for e-commerce recommendation, influencer audience modeling, and creator persona analysis |
| **Total** | **100** | **70** | |

---

## 电商/内容治理相关性 / E-commerce & Governance Relevance

PHF 框架与电商达人生态系统直接对应：
- **Practice** = 达人的历史内容（短视频、直播、商品推广）
- **Habitus** = 达人稳定的内容风格和受众偏好
- **Field** = 同类型达人的群体规律（美妆类、3C类等）

应用场景：
1. **达人内容推荐**：基于 Habitus 理解达人的受众偏好，提供更精准的内容推荐策略
2. **用户个性化**：基于 Field 的群体知识为冷启动用户提供初始个性化
3. **违规倾向建模**：通过 Habitus 分析达人的历史行为模式，识别潜在违规风险

PHF maps directly to e-commerce creator ecosystems: Practice = creator content history, Habitus = stable content style/audience preference, Field = creator group patterns (beauty, tech, etc.). Applications: influencer content recommendation, cold-start personalization, and violation tendency modeling.
