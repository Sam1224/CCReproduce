## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **标题** | CFALR: Collaborative Filtering-Augmented Large Language Model for Personalized Fashion Outfit Recommendation |
| **作者** | Yujuan Ding, Junrong Liao, Yunshan Ma, Yi Bin, Wenqi Fan, Tat-Seng Chua, Qing Li |
| **机构** | The Hong Kong Polytechnic University, UESTC, Singapore Management University, Tongji University, NUS |
| **arXiv** | [2606.13001](https://arxiv.org/abs/2606.13001) |
| **发布日期** | 2026-06-12 |
| **领域标签** | 电商、推荐、穿搭/组合推荐、LLM、协同过滤、多模态 |

---

## 方法概述

**Story Arc**：穿搭推荐既要个性化又要兼容性，且组合空间巨大。传统 CF 在稀疏时失效，模板法受限于固定结构；LLM 语义强但与交互 ID 空间不匹配。CFALR 的路径：(1) 把 outfit completion 改写为 **Personalized Fill-In-The-Blank (P-FITB)** 任务，让 LLM "读懂"穿搭场景；(2) 把 **CF 向量与多模态 item 特征通过投影层映射为单 token** 注入 LLM；(3) 推理时用 **CF-augmented 生成机制**在输出层融合协同信号与语言模型分布，兼顾个性化与语义理解。

**Innovation**：区别于 BPR/NCF 等 CF-only 方案和 FashionGPT 等纯 LLM 方案，CFALR 首次在 outfit-level 组合推荐中实现显式 CF+LLM 特征 token 融合，并在稀疏设置下证明两者互补。

---

## 关键指标

| 数据集 | 指标（P-FITB Acc.） | CFALR | 最强 Baseline (Bundle-MLLM) |
|--------|-------------------|-------|--------------------------|
| Polyvore（1/4 difficulty） | P-FITB Acc | **0.6498** | 0.4775 |
| Polyvore（1/10） | P-FITB Acc | **0.3957** | 0.2519 |
| Polyvore（1/20） | P-FITB Acc | **0.2459** | 0.1459 |
| IQON（1/4） | P-FITB Acc | **0.6103** | — |

---

## 评分

| 维度 | 得分 | 说明 |
|------|------|------|
| 方法创新性 | 23/30 | P-FITB task + CF token 融合有新意 |
| 实验指标 | 12/15 | 大幅超越 baseline，两数据集验证 |
| 实验质量 | 12/15 | Ablation 充分，但无线上验证 |
| 效率 | 5/10 | LLM + CF 双模块推理开销较高 |
| 泛化性 | 3/5 | 穿搭推荐为主，其他组合推荐场景待验证 |
| 领域相关性 | 23/25 | 与电商组合推荐/内容生态（穿搭）直接相关 |
| **Total** | **78/100** | 离线指标提升显著，待线上验证 |
