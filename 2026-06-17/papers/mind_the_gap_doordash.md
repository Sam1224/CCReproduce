# Mind the Gap: Bridging Behavioral Silos with LLMs in Multi-Vertical Recommendations

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | Mind the Gap: Bridging Behavioral Silos with LLMs in Multi-Vertical Recommendations |
| **Authors** | Nimesh Sinha, Raghav Saboo, Martin Wang, Sudeep Das |
| **Affiliations** | DoorDash Inc. |
| **arXiv** | [2606.06779](https://arxiv.org/abs/2606.06779) |
| **Submitted** | 2026-06-04 (presented at RecSys 2025) |
| **Domain Tags** | multi-vertical recommendation, LLM, RAG, MTL, DoorDash |

---

## 方法概述 / Method Summary

DoorDash 等多业态平台存在严重的"行为孤岛"问题——餐厅业务积累了丰富用户数据，而商超、杂货等新业态数据稀疏。本文提出利用 LLM 进行生成式推断：基于层次化 RAG 流水线从餐厅历史订单和搜索词提取多级分类特征，将其作为用户偏好信号注入 Multi-Task Learning（MTL）主排序模型，实现跨业态知识迁移。

**Story arc**: 多业态平台数据孤岛制约新业态推荐质量 → 分层 RAG + LLM 生成用户跨域分类特征，注入 MTL 排序。

---

## 评分 / Scoring

| 维度 | 分数 | 满分 | 说明 |
|------|------|------|------|
| Innovation | 20 | 30 | 分层 RAG + MTL 组合，工程为主 |
| Experimental SOTA delta | 9 | 15 | 在线 A/B 结果提及但未公开具体数值 |
| Experimental quality / ablations | 11 | 15 | DoorDash 生产验证 |
| Efficiency | 7 | 10 | LLM 推断离线化 |
| Generalization | 3 | 5 | DoorDash 特定，多业态泛化性有限 |
| Domain relevance | 17 | 25 | 多业态推荐，与电商内容治理间接相关 |
| **Total** | **68** | **100** | 工业落地合理，但创新度中等，具体数值未公开 |
