# When Rules Fall Short: Agent-Driven Discovery of Emerging Content Issues in Short Video Platforms

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | When Rules Fall Short: Agent-Driven Discovery of Emerging Content Issues in Short Video Platforms |
| **Authors** | Chenghui Yu, Zixuan Wang, Hongwei Wang, Bingfeng Deng, Junwen Chen, Zhuolin Hao, Hongyu Xiong |
| **Affiliation** | TikTok Inc. |
| **arXiv** | https://arxiv.org/abs/2601.11634 |
| **Submitted** | January 2026 |
| **Domain** | Short Video Content Governance · Agent · Clustering · Data Labeling · Policy Iteration |
| **Code** | `code/AgentIssueDiscovery/` |

---

## 方法概述 / Method Summary

短视频平台每天出现大量新兴内容问题，超出现有标注策略的覆盖范围（policy gap）。传统的"人工发现 → 补充策略"流程耗时长、效率低，导致问题视频在被治理之前积累了大量曝光。

本文提出 **Agent-Driven Issue Discovery** 系统，包含三个核心阶段：

1. **召回 (Recall):** 多模态 LLM Agent 分析视频内容（视觉 + 音频 + OCR），识别可能包含潜在新问题的视频候选集。Agent 结合已有策略库进行对比推理，标记那些"策略未能明确覆盖但行为异常"的视频。

2. **两阶段聚类 (Two-Stage Clustering):** 
   - 第一阶段：对候选视频做粗粒度聚类，将同一问题变体（variant）归为一簇
   - 第二阶段：在每个粗粒度簇内做细粒度聚类，区分"已知问题的新变体"与"全新子问题"
   
3. **策略生成 (Policy Generation):** 从聚类簇中自动生成新的标注策略描述，供人工审核后纳入策略库。

---

## 故事弧线 / Story Arc

> 现有策略无法追上快速涌现的新内容问题 → 设计多模态 LLM Agent 自动发现新问题 + 两阶段聚类区分变体与新问题 → 大幅提升发现效率，显著减少违规视频曝光

---

## 创新分析 / Innovation Analysis

**与前人工作的差异：**
- 现有内容治理研究聚焦于对已知问题的分类/检测，本文**首次**提出系统化的新兴问题自动发现框架
- 两阶段聚类策略能区分"变体"与"新问题"，避免把所有异常内容混为一谈
- 基于多模态 LLM 的召回比关键词/规则匹配更鲁棒，能捕捉视觉层面的异常
- 自动策略生成闭环大幅加速标注策略迭代，属于**工业系统创新**

**可行性：** 已在 TikTok 平台实际部署，有真实系统指标背书。

---

## 关键指标 / Key Metrics

| Dataset/Metric | Method | Value |
|----------------|--------|-------|
| Issue Discovery | F1 Score improvement (vs. manual) | **+20%** |
| Platform Impact | Reduction in problematic video views | **~15%** |
| Operational | Time cost for policy iteration | 显著降低（大幅） |

---

## 评分 / Scoring

| 维度 | 得分 | 满分 | 说明 |
|------|------|------|------|
| Innovation | 25 | 30 | 首次系统化新兴问题自动发现，两阶段聚类新颖 |
| Experimental SOTA delta | 13 | 15 | F1 +20%，违规曝光 -15%，实部署数据 |
| Experimental quality | 13 | 15 | 工业部署 A/B 验证，真实指标可信 |
| Efficiency | 9 | 10 | 大幅减少人工时间成本 |
| Generalization | 4 | 5 | 适用于所有短视频平台 |
| Domain relevance | 25 | 25 | 完全命中：短视频平台内容治理 + 达人监管 + 自动标注 |
| **Total** | **89** | **100** | |

---

## 代码复现说明

完整 PyTorch 复现位于 `code/AgentIssueDiscovery/`

核心组件：
- `agent/`: LLM Agent 召回模块（含多模态提示模板）
- `clustering/`: 两阶段聚类（HDBSCAN + 对比嵌入）
- `policy_gen/`: 策略描述自动生成
- `pipeline.py`: 端到端 pipeline
- `train.py` / `eval.py`: 训练与评测
