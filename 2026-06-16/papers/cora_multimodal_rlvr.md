# CORA: Multimodal RLVR Reasoning Alignment

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **标题** | CORA: Analyzing and Bridging Thinking-Answer Gap in Multimodal RLVR via Consistency-Oriented Reasoning Alignment |
| **作者** | (作者信息待确认) |
| **机构** | (待确认) |
| **链接** | https://arxiv.org/abs/2606.14691 |
| **arXiv ID** | 2606.14691 |
| **提交日期** | ~June 13, 2026 ★ (在June 16 Monday arXiv listing中) |
| **Bucket** | WEAK |

---

## 方法概述 / Method Overview

**中文：**  
CORA针对多模态强化学习（RLVR）训练中存在的**思维-答案语义不一致**（Thinking-Answer Inconsistency）问题提出解决方案。现有多模态RLVR方法主要关注视觉覆盖和幻觉减少，却忽略了推理过程（thinking chain）与最终答案（answer）之间的语义矛盾。CORA通过引入**一致性导向推理对齐（Consistency-Oriented Reasoning Alignment）**，设计了一个轻量级即插即用的一致性奖励模型，通过RL奖励信号鼓励模型在推理轨迹和最终答案之间保持语义一致。

**English:**  
CORA addresses the thinking-answer semantic inconsistency problem in multimodal RLVR training. Prior RLVR methods focus on visual coverage and hallucination reduction but overlook semantic conflicts between reasoning chains and final answers. CORA introduces a lightweight plug-in consistency reward model (CRM) that penalizes semantic inconsistency between thinking and answer, improving multimodal reasoning fidelity.

---

## 故事弧线 / Story Arc

**现有方法不足 →** 多模态RLVR训练的思维链与最终答案经常语义矛盾（"想一套、说一套"），影响推理可靠性。  
**本文设计 →** 即插即用一致性奖励模型（CRM），量化思维-答案语义一致性并作为RLVR额外奖励信号，无需修改基础训练框架。

---

## 创新性 / Innovation

1. **思维-答案一致性量化**：首次明确分析并量化多模态RLVR中thinking-answer semantic gap问题。
2. **即插即用设计**：CRM无需修改现有RLVR框架，直接叠加一致性奖励。
3. **适用多模态场景**：对电商产品理解、内容审核等需要精确推理的多模态场景有参考价值。

---

## 评分 / Scoring

| 维度 | 分数 | 满分 | 说明 |
|------|------|------|------|
| 创新性 (Innovation) | 18 | 30 | 问题识别有价值，解决方案较直接 |
| 实验SOTA增量 (SOTA Delta) | 11 | 15 | 多个多模态基准上有提升 |
| 实验质量/消融 (Exp Quality) | 10 | 15 | 消融实验合理 |
| 效率 (Efficiency) | 7 | 10 | 即插即用开销低 |
| 泛化性 (Generalization) | 3 | 5 | 通用RLVR训练改进 |
| 领域相关性 (Domain Relevance) | 12 | 25 | 多模态推理通用，间接相关 |
| **Total** | **61** | **100** | |

---

## 参考链接

- arXiv: https://arxiv.org/abs/2606.14691
