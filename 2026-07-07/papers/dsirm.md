# DSIRM: Learning Query-Bridged Discrete Semantic Identifiers for E-commerce Relevance Modeling

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **标题** | DSIRM: Learning Query-Bridged Discrete Semantic Identifiers for E-commerce Relevance Modeling |
| **作者** | Zhenghao Liu et al. |
| **机构** | Taobao & Tmall Group, Alibaba |
| **arXiv** | https://arxiv.org/abs/2606.04374 |
| **发表日期** | 2026-06-03 |
| **标签** | E-commerce Search · Semantic ID · Relevance Modeling · LLM · Alibaba |

---

## 故事弧 / Story Arc

**现有困境：** 电商搜索相关性建模中，离散语义 ID（SID）已被采用但依赖无监督量化——码本分区无法感知 query 相关性，导致"同款不同叫法"、长尾 query、意图歧义等问题难以解决。连续 embedding 对细粒度属性区分效果也有限。

**我们的设计：** DSIRM 通过 query-bridged 对比量化（item 侧）和生成式 LLM SID 预测（query 侧）构建双路互补体系，使 SID 具备相关感知能力。

---

## 方法摘要 / Method Overview

### 双路设计

**Item 侧：Query-Bridged Contrastive Quantization（CBCQ）**
- 在残差量化（Residual Quantization）过程中注入 query-item 交互监督信号
- 码本分区从无监督聚类变为"相关感知语义区间"
- 同一 query 下相关 item 的 SID 前缀趋同，不相关 item 前缀分化

**Query 侧：生成式 LLM 预测 item SID**
- LLM 从 query 文本直接预测目标 item 的 SID
- 解决长尾 query 意图表达不充分的问题
- 隐式建模 query 意图 → 商品属性 映射

**层次前缀匹配**
- query SID（LLM 预测）与 item SID（CBCQ 量化）做层次前缀匹配
- 生成判别性特征，与密集检索信号互补
- 高效可扩展，适合亿级商品库

---

## 关键指标 / Key Metrics

层次前缀匹配特征与密集检索信号互补，在阿里巴巴生产规模数据集上带来显著相关性提升；目前已在阿里电商搜索系统线上部署（具体数值见论文表格）。

---

## 评分 / Score

| 维度 | 得分 | 最高 |
|------|------|------|
| Innovation | 20 | 30 |
| Experimental SOTA delta | 11 | 15 |
| Experimental quality / ablations | 11 | 15 |
| Efficiency | 7 | 10 |
| Generalization | 3 | 5 |
| Domain relevance (ecom + governance) | 21 | 25 |
| **Total** | **73** | **100** |

**评分理由：** 阿里电商搜索核心技术，双路互补设计解决真实痛点；具体数字未完全公开是扣分原因；任务垂直性导致泛化有限。
