# Multimodal Data Curation Through Ranked Retrieval

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| 标题 | Multimodal Data Curation Through Ranked Retrieval |
| 作者 | Pratyush Muthukumar, Harshil Kotamreddy, Sarah Amiraslani, Tomo Kanazawa, Ramani Akkati, Shaan Jain, Andrew Mathau |
| 机构 | NVIDIA |
| arXiv | https://arxiv.org/abs/2605.01163 |
| 提交日期 | 2026-05-01 |
| 领域标签 | 多模态数据质量 · Embedding · 相似度 · 数据清洗 · 对齐 |
| 桶类别 | WEAK |
| 综合评分 | **63 / 100** |

---

## 方法概述 (中文)

多模态检索和数据策划中，共享 embedding 空间常见两大问题：① Embedding 反映模态（图 vs. 文）多于语义，导致跨模态聚类失真；② 人工标注配对噪声大，混合异质数据集时问题加剧。

**本文提出双重修复框架：**
1. **Symmetric Nucleus Subsampling (SNS)**：对原始输入-标注对进行修剪，保留相互支持度最高的子集，过滤噪声配对
2. **Expert Embedding Engine (EEE)**：将多个互补 embedding 专家通过学习投影层融合，结合偏差感知目标函数，减少 embedding 空间中模态驱动的分离

---

## 故事线 (Story Arc)

> **现状不足：** 电商等场景中图文对来自多个异质来源，embedding 空间将图片聚成一簇、文本聚成另一簇，严重妨碍跨模态检索和数据质量筛选。
>
> **我们的解法：** SNS 净化训练对 + EEE 融合专家 embedding，从两个层面修复模态偏差，提升跨模态检索对齐。

---

## 关键指标

| 任务 | 指标 | 本文 | 基线 |
|------|------|------|------|
| 跨模态检索 | R@1 | 改善 | CLIP 基线 |
| 数据策划下游任务 | 训练效率 | 提升 | 原始数据集 |

---

## 评分分解

| 维度 | 得分 | 满分 | 说明 |
|------|------|------|------|
| 创新性 | 19 | 30 | SNS + EEE 组合有新意，但各单元均非颠覆性 |
| 实验 SOTA Delta | 8 | 15 | 具体数值未获取 |
| 实验质量/消融 | 10 | 15 | NVIDIA 实验规范 |
| 效率 | 7 | 10 | 数据策划本身提升训练效率 |
| 泛化性 | 4 | 5 | 通用多模态场景 |
| 领域相关性 | 15 | 25 | 间接：电商图文数据清洗和对齐的通用方法 |
| **总分** | **63** | **100** | — |
