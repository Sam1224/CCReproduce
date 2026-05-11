# PREFER: Personalized Review Summarization with Online Preference Learning

## 基本信息 / Basic Info

| 字段 | 内容 |
|---|---|
| **标题** | PREFER: Personalized Review Summarization with Online Preference Learning |
| **arXiv ID** | [2605.05911](https://arxiv.org/abs/2605.05911) |
| **提交日期** | 2026-05-07 |
| **作者** | Millend Roy, Agostino Capponi, Vineet Goyal |
| **机构** | Columbia University |
| **领域桶** | STRONG |
| **综合评分** | **68 / 100** |

---

## 方法概述 (Chinese)

电商平台上的商品评论是用户购买决策的重要参考，但海量评论信息过载问题严峻：用户关注不同的产品特性（如耐用性、外观、性价比），而现有摘要系统生成的是千篇一律的"通用摘要"，无法匹配个体偏好，且偏好会随时间演化。

PREFER（**P**ersonalized **R**eview Summarization with Onlin**e** Pre**fer**ence Learning）提出在线偏好学习框架，解决**未知潜在偏好（unknown latent preferences）**问题：

1. **在线学习机制**：系统通过已生成摘要中直接获取的用户反馈，迭代精化对用户偏好的理解；
2. **个性化摘要生成**：为每位用户生成针对其关注维度的定制化摘要；
3. **偏好演化建模**：框架支持随用户-系统交互而动态更新的偏好表示。

在 Amazon Reviews'23 数据集的受控仿真中验证，在线偏好学习显著提升了与目标用户兴趣的对齐度，同时保持摘要质量。

## Method Overview (English)

PREFER addresses the cold-start problem of unknown user preferences in e-commerce review summarization via an online learning framework. The system iteratively refines its understanding of user preferences by incorporating feedback from generated summaries. Evaluated on Amazon Reviews'23 via controlled simulation; online preference learning improves alignment with target user interests while maintaining summary quality.

---

## Story Arc

**电商评论摘要系统生成通用静态摘要，忽视用户个体偏好的差异性和时变性 → PREFER 通过在线偏好学习框架迭代地从摘要反馈中精化用户偏好模型，为每位用户生成个性化摘要。**

> A user who cares about battery life but sees a camera review gets a mis-targeted summary. PREFER learns over time what each user actually cares about — from the feedback implicit in how they respond to summaries.

---

## 创新性分析

1. **在线学习范式**：从离线偏好建模转向在线迭代精化，更接近实际部署场景；
2. **摘要反馈闭环**：从摘要本身获取反馈（而非评分或点击）是独特的信号来源；
3. **偏好演化支持**：动态偏好表示比静态个性化更贴近用户行为规律。

**局限性**：基于仿真评估，缺乏真实用户实验；偏好反馈的获取假设较理想化。

**与电商内容生态的关联**：商品摘要是核心 UGC 内容形态，个性化摘要可提升转化率和用户体验。

---

## 关键指标 / Key Metrics

| 数据集 | 指标 | PREFER | 静态摘要基线 |
|---|---|---|---|
| Amazon Reviews'23（受控仿真） | 偏好对齐度 | **显著提升** | 较低 |
| Amazon Reviews'23 | 摘要质量（ROUGE等） | 维持 | 维持 |

---

## 评分详情 / Scoring Breakdown

| 维度 | 满分 | 得分 | 说明 |
|---|---|---|---|
| 创新性 (Innovation) | 30 | 18 | 在线偏好学习框架是有价值创新，但偏好建模本身非首次 |
| 实验SOTA增益 (SOTA delta) | 15 | 9 | 仿真实验，对齐度提升，缺乏真实用户验证 |
| 实验质量与消融 (Quality) | 15 | 9 | 受控仿真，方法严谨但缺真实实验 |
| 效率 (Efficiency) | 10 | 7 | 在线学习开销适中 |
| 泛化性 (Generalization) | 5 | 3 | 单数据集，仿真场景 |
| 领域相关性 (Domain) | 25 | 22 | 直接电商评论场景，个性化内容生成 |
| **总分** | **100** | **68** | — |

---

## 链接 / Links

- 论文: https://arxiv.org/abs/2605.05911
- HTML版: https://arxiv.org/html/2605.05911
