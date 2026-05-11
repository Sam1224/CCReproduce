## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **标题** | Multimodal Data Curation Through Ranked Retrieval |
| **arXiv ID** | [2605.01163](https://arxiv.org/abs/2605.01163) |
| **提交日期** | 2026-05（May 2026） |
| **作者** | Pratyush Muthukumar, Harshil Kotamreddy, Sarah Amiraslani, Tomo Kanazawa, Ramani Akkati, Shaan Jain, Andrew Mathau |
| **机构** | 未完全公开（工业研究团队）|
| **领域** | Multimodal Embedding · Data Curation · Cross-Modal Retrieval · Data Quality |
| **Bucket** | STRONG |

---

## 方法概述 / Method Summary

本文针对多模态共享嵌入空间中两个核心问题提出解决方案：（1）模态偏差：嵌入反映模态类型而非语义内容，导致跨模态检索失效；（2）标注噪声：混合多异构数据集时成对监督信号噪声放大。

**核心组件：**

1. **SNS（Symmetric Nucleus Subsampling，对称核采样）**：对原始输入和标注进行修剪，保留彼此最能相互支撑的部分，精炼训练对。通过"核"机制去除不相关内容，减少噪声标注影响。

2. **EEE（Expert Embedding Engine，专家嵌入引擎）**：用可学习投影网络组合多个互补嵌入专家（各专家在不同模态/任务上有专长），同时引入偏差感知目标函数，显式减少嵌入空间中的模态驱动分离。

**训练流程：** SNS 清洗训练对 → EEE 组合专家嵌入 → 对比学习优化跨模态对齐。

### Story Arc
> **多模态嵌入空间中模态偏差和标注噪声导致跨模态检索质量下降** → SNS 从训练数据层面净化配对质量，EEE 从模型层面融合互补专家并抑制模态分离偏差，双管齐下提升多模态数据策展和检索质量。

---

## 创新性分析 / Innovation Analysis

**与先前工作对比：**
- CLIP 系列：对比学习多模态嵌入，但未解决模态偏差和配对噪声
- DataComp、MMC4 等数据策展工作：关注数据筛选但未显式建模模态分离
- 本文：在训练对级别（SNS）和模型级别（EEE）双重干预，且 EEE 可组合多个现有嵌入专家

**关键创新：**
1. SNS 提供了一种通用的配对精炼机制，适用于任何多模态对比学习
2. EEE 的专家组合框架可即插即用地利用已有预训练嵌入模型
3. 偏差感知目标函数显式建模并减少模态分离

**电商应用价值：** 直接适用于商品图文配对数据清洗（海量 SKU 的图片-标题-描述三模态对齐质量提升）。

---

## 关键指标 / Key Metrics

| 任务 | 指标 | SNS+EEE | Baseline |
|------|------|---------|---------|
| Cross-modal Retrieval (multi-dataset) | Recall@K | 显著提升 | 标准 CLIP/OpenCLIP |
| Pair Consistency | Symmetric Consistency Score | 提升 | 原始配对 |

---

## 评分 / Scoring

| 维度 | 得分 | 说明 |
|------|------|------|
| Innovation | 18/30 | 双层干预框架较为新颖，但问题本身有先例 |
| SOTA Delta | 9/15 | 多数据集上有提升，细节待补充 |
| Exp Quality / Ablations | 10/15 | 分析模态偏差和配对质量的多维评估 |
| Efficiency | 6/10 | 投影网络轻量，但专家组合有额外推理成本 |
| Generalization | 3/5 | 多异构数据集测试 |
| Domain Relevance | 17/25 | 电商商品多模态嵌入数据质量直接受益 |
| **总分** | **63/100** | Feishu 推送 |
