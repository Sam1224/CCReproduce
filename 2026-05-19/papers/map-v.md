# MAP-V: Transparent and Controllable Recommendation Filtering via Multimodal Multi-Agent

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | Transparent and Controllable Recommendation Filtering via Multimodal Multi-Agent Collaboration |
| **Authors** | Chi Zhang, Zhipeng Xu, Jiahao Liu, Dongsheng Li, Hansu Gu, Peng Zhang, Ning Gu, Tun Lu |
| **Affiliation** | Fudan University; Microsoft Research Asia |
| **arXiv** | https://arxiv.org/abs/2604.17459 |
| **Submitted** | April 2026 |
| **Domain** | Recommendation Filtering · Multimodal · Multi-Agent · Content Governance · Preference Graph |

---

## 方法概述 / Method Summary

个性化推荐系统能高效发现内容，但会给用户推送不想要的内容。现有 LLM 方法存在两大瓶颈：(1) 缺乏多模态感知，无法识别视觉层面的不当内容；(2) "过度关联"（over-association）问题：把用户对某特定内容的不喜欢错误地泛化，导致误屏蔽良性内容。

**MAP-V（Multimodal Agent Pipeline for Visual filtering）** 提出：
- **可编辑偏好图（Editable Preference Graph）：** 将用户偏好结构化存储，支持细粒度更新和可解释的过滤决策
- **四智能体流水线：**
  - **Intent Parser：** 解析用户的实际意图，区分"不喜欢"的精确含义
  - **Judge Agent：** 基于偏好图做初步判断
  - **Dispute Agent：** 对争议情况进行二次审查，减少误判
  - **RAH Agent（Retrieval-Augmented History）：** 结合历史行为辅助决策
- **端云协同架构：** 轻量端侧过滤 + 云端精判，平衡延迟与准确率

---

## 故事弧线 / Story Arc

> 推荐系统过滤层缺乏多模态感知且存在过度关联误判问题 → MAP-V 用可编辑偏好图+四 Agent 协作实现精准意图对齐 → 透明可解释的多模态推荐过滤

---

## 创新分析 / Innovation Analysis

- 可编辑偏好图是一个新颖的用户意图建模结构，使过滤决策可追溯
- Dispute Agent 专门处理边界案例，减少过度关联误判，是对 LLM 判断能力局限的有效补充
- 端云协同设计兼顾实时性和准确率，具有工程实用价值

---

## 关键指标 / Key Metrics

| Task | Metric | MAP-V vs Baselines |
|------|--------|-------------------|
| Recommendation Filtering | Precision / Recall | Outperforms LLM-only baselines |
| Over-association Rate | False block rate | Significant reduction |
| Multimodal Filtering | Visual inappropriate content detection | vs. text-only methods: +↑ |

---

## 评分 / Scoring

| 维度 | 得分 | 满分 | 说明 |
|------|------|------|------|
| Innovation | 22 | 30 | 偏好图+四 Agent+端云协同，设计完整 |
| Experimental SOTA delta | 10 | 15 | 超过 LLM 基线，减少误判 |
| Experimental quality | 10 | 15 | 端云协同验证，ablation 合理 |
| Efficiency | 7 | 10 | 端云协同注重效率 |
| Generalization | 3 | 5 | 通用推荐场景 |
| Domain relevance | 20 | 25 | 推荐内容过滤+治理直接相关 |
| **Total** | **72** | **100** | |

