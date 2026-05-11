# Negative Data Mining for Contrastive Learning in Dense Retrieval at IKEA.com

- **arXiv**: https://arxiv.org/abs/2605.00353
- **作者**: Eva Agapaki, Amritpal Singh Gill (IKEA.com)
- **领域标签**: `Dense Retrieval` `电商-搜索` `LLM-as-Judge` `Hard Negatives`

## 方法概述
在 IKEA 商品检索 (late-interaction 双塔架构) 上系统比较 contrastive 训练的负样本策略：
1. **Structured negative sampling**: 利用商品分类树 + 商品属性挑语义难负样本；
2. **LLM-as-Judge** 给每个 query 对全候选商品打相关分，替代稀疏人工标注。

## 故事
Dense retrieval 训练 → 负样本质量决定上限 → 行业内主要靠人标 + 随机采样，质量有限 → 用商品 taxonomy + LLM 评分构造结构化难负 + 全候选打分。

## 创新性分析
- 把 LLM-as-Judge 系统化用作 retrieval 训练数据生成器，落地视角清晰；
- "structured negative" 的 taxonomy + attribute 用法在工业上并不少见，但作为系统化报告少；
- **诚实呈现 A/B 失利**：结论认为问题不在召回质量，而在用户搜索行为 (67% 热门 query 零点击率 >50%)，这种"负面结果 + 归因"是工业研究里很有价值的部分。

## 关键指标
- 离线 (Canada market real query): **类目准确率 +2.6%** 平均；
- 在线 A/B (long-tail query)：**用户参与指标无显著差异**, p > 0.05.

## 评分
| 维度 | 分 / 满分 | 理由 |
|------|----------|------|
| 创新性 | 15 / 30 | 工程组合，方法层面增量 |
| 实验指标 | 7 / 15 | A/B 未显著，但诚实 |
| 实验质量 | 12 / 15 | 真实工业 A/B + 行为分析 |
| 效率 | 6 / 10 | LLM 全候选评分代价不低 |
| 泛化性 | 3 / 5 | IKEA-specific 但方法可迁 |
| 相关性 | 25 / 25 | 真实电商搜索负样本工程 |
| **合计** | **68** |
