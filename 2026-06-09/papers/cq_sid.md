# CQ-SID + EG-GRPO: Efficient Generative Retrieval for E-commerce Search with Semantic Cluster IDs and Expert-Guided RL

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | Efficient Generative Retrieval for E-commerce Search with Semantic Cluster IDs and Expert-Guided RL |
| **Authors** | Jianbo Zhu, et al. (8 authors) |
| **Affiliations** | (Chinese e-commerce platform) |
| **arXiv** | [2605.14434](https://arxiv.org/abs/2605.14434) |
| **Submitted** | 2026-05-14 |
| **Keywords** | 电商搜索、生成式召回、语义聚类 ID、强化学习、RL 对齐 |
| **Bucket** | STRONG |
| **Code** | `code/CQ-SID/` (本仓库 toy 复现) |

---

## 方法概述 / Method Summary

生成式召回（Generative Retrieval, GR）将多阶段检索流程合并为单一端到端模型，但在工业电商搜索中面临三大挑战：**海量动态商品目录**、**严格的延迟要求**、**与下游排序目标的对齐问题**。本文提出 CQ-SID + EG-GRPO 框架，将 GR 定位为**召回阶段的补充**（而非完全替代传统检索），具体包含：

1. **CQ-SID（Category-and-Query constrained Semantic ID）**：
   - 利用类目感知（category-aware）对比学习和查询-商品对比学习；
   - 结合 Residual Quantized VAE（RQ-VAE）将商品编码为层次语义聚类标识符；
   - 显著减少 beam search 复杂度（相比直接用商品 ID）。

2. **EG-GRPO（Expert-Guided Group Relative Policy Optimization）**：
   - 引入强化学习来对齐生成式召回与下游排序目标；
   - 专家引导（expert-guided）设计解决奖励稀疏和反馈延迟问题；
   - 在 GRPO 框架下通过组相对策略优化稳定训练。

---

## 故事弧 / Story Arc

> **现状不足** → **提出方案**

传统电商搜索使用多阶段检索（粗排 → 精排），GR 理论上可以合并流程，但在工业环境中面临三大障碍：（1）动态目录规模庞大导致 beam search 复杂度爆炸；（2）GR 的优化目标（token 预测 likelihood）与下游点击/转化目标存在偏差；（3）端到端替代方案延迟太高。

CQ-SID + EG-GRPO 将 GR 定位为"召回补充"：CQ-SID 通过结构化层次编码大幅压缩搜索空间，EG-GRPO 通过专家引导 RL 实现目标对齐，使得 GR 既高效又与业务指标对齐。

---

## 创新性分析 / Innovation

| 维度 | 分析 |
|------|------|
| vs. 纯 GR（端到端） | 明确定位为召回补充，降低了延迟和容错风险 |
| vs. 标准 RQ-VAE 语义 ID | 引入类目+查询双重约束，提升业务相关性 |
| vs. 标准 GRPO | 专家引导奖励设计解决了纯在线奖励信号稀疏的问题 |
| 实用性 | 直接针对工业电商三大痛点，落地导向明确 |

**可行性评估**：高。RQ-VAE 在语义 ID 领域已有成功案例（SID-Coord、FLUID），GRPO 是成熟 RL 框架。

---

## 关键指标 / Key Metrics

| 数据集/场景 | 指标 | CQ-SID+EG-GRPO | Baseline GR |
|-------------|------|----------------|-------------|
| 工业电商搜索召回 | Recall@50 | +significant | GR standard |
| Beam search 复杂度 | FLOPs | 显著降低 | — |
| 下游排序对齐 | NDCG | 优于普通 GR | — |

---

## 评分明细 / Score Breakdown

| 维度 | 得分 | 满分 | 说明 |
|------|------|------|------|
| 方法创新性 | 23 | 30 | 双约束 SID + 专家引导 RL 新颖；但组合创新多于原创概念 |
| 实验指标 SOTA | 12 | 15 | 工业数据结果显著，但无公开基准对比 |
| 实验质量/消融 | 12 | 15 | CQ-SID 和 EG-GRPO 各自消融充分 |
| 方法效率 | 9 | 10 | beam search 复杂度大幅降低是核心贡献之一 |
| 方法泛化性 | 3 | 5 | 专注电商搜索场景 |
| 论文相关性 | 25 | 25 | 直接命中电商搜索召回场景，与核心域高度契合 |
| **Total** | **84** | **100** | 工业电商搜索核心问题，有效解决 GR 落地三大瓶颈 |
