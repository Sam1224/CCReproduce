# Taiji: Pareto Optimal Policy Optimization with Semantics-IDs Trade-off for Industrial LLM-Enhanced Recommendation

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| 标题 | Taiji: Pareto Optimal Policy Optimization with Semantics-IDs Trade-off for Industrial LLM-Enhanced Recommendation |
| 作者 | Yuecheng Li, Zeyu Song, Jing Yao, Chi Lu, Peng Jiang, Kun Gai |
| 机构 | Kuaishou Technology（快手科技）|
| arXiv | https://arxiv.org/abs/2606.03866 |
| 提交日期 | 2026-06-03 |
| 领域标签 | 推荐系统 · LLM增强推荐 · 强化学习 · 工业部署 · 广告 |
| 桶类别 | STRONG |
| 综合评分 | **84 / 100** |
| 代码复现 | `code/Taiji/` |

---

## 方法概述 (中文)

LLM 在工业推荐系统中的落地面临两大核心困境：（1）**SFT瓶颈**：直接监督微调使 LLM 记忆在线 ID 协同特征，但在线用户偏好随时变化，导致 LLM 泛化能力受损；（2）**RL对齐问题**：强化学习训练时如何在 LLM 语义世界知识与线上协同 ID 特征之间取得最优平衡，缺乏理论保证。

**Taiji** 提出两大技术：

1. **逆向推理 CoT 数据生成（Reverse-Engineered CoT）**：避免直接让 LLM 记忆 ID 序列，而是从用户行为序列反推"用户为什么会有这些偏好"，生成高质量、领域特化的推理链（Chain-of-Thought）数据用于 SFT，使 LLM 学习语义推理而非 ID 记忆。

2. **帕累托最优策略优化（POPO）**：在 RL 阶段设计多目标奖励框架，将语义奖励（LLM 的世界知识）和 ID 奖励（协同过滤的用户偏好信号）视为两个目标，动态调整跨域奖励权重，理论上保证达到帕累托最优的语义-ID 权衡，避免任一目标过度主导导致的退化。

**在线部署**：Taiji 已于 2026 年 5 月部署至快手广告平台，日服务用户超 4 亿，在线 A/B 实验显示广告主价值（ADVV）提升 **+2.83%**，整体营收提升 **+3.30%**。

---

## 故事线 (Story Arc)

> **现状不足：** LLM 直接参与推荐有两道墙——SFT 阶段 LLM 死记在线 ID 而失去泛化能力；RL 阶段语义奖励与协同 ID 奖励相互拉扯，无法保证最优权衡。
>
> **我们的解法：** Taiji 用"逆向 CoT 数据"解决 SFT 瓶颈（让 LLM 学理由而非记 ID），再用 POPO 在 RL 阶段用帕累托理论保证语义-ID 最优权衡，让工业级 LLM 推荐在 4 亿用户规模安全落地并创造实际商业收益。

---

## 创新点分析

| 维度 | 描述 |
|------|------|
| 核心创新 | POPO：首次将帕累托最优理论引入 LLM 推荐 RL 对齐，动态权衡语义-ID 奖励 |
| SFT 突破 | 逆向工程 CoT 数据生成，让 LLM 学因果推理而非 ID 记忆，突破 SFT 瓶颈 |
| vs. 先前工作 | 传统 LLM4Rec 直接 SFT 或固定权重 RL；Taiji 从理论层面保证多目标奖励最优 |
| 规模 | 日活 4 亿用户，已在快手广告真实线上部署（非实验系统） |
| 局限 | 主要场景为广告推荐；LLM 推理延迟管理未详述；CoT 数据生成成本较高 |

---

## 关键指标

| 实验 | 数据集/场景 | 指标 | Taiji | Baseline |
|------|------------|------|-------|---------|
| 线上 A/B | 快手广告平台（4亿日活） | ADVV | +2.83% | — |
| 线上 A/B | 快手广告平台 | 整体营收 | +3.30% | — |
| 离线评估 | 工业级推荐数据集 | 未公开具体数值（论文声明显著超越 baseline） | — | — |

---

## 评分分解

| 维度 | 分数 | 满分 | 说明 |
|------|------|------|------|
| Innovation | 24 | 30 | POPO 将帕累托理论引入 LLM RL 对齐，方法论新颖；逆向 CoT 数据生成是务实创新 |
| Experimental SOTA delta | 13 | 15 | 线上 4 亿用户 A/B 测试，营收 +3.30% 是大规模工业级证明 |
| Experimental quality | 12 | 15 | 离线+线上双验证，但离线数据集细节未完全公开 |
| Efficiency | 7 | 10 | POPO 动态权重调整计算高效；LLM 离线推理减轻在线延迟压力 |
| Generalization | 4 | 5 | 可迁移到其他推荐场景，但目前主要验证广告场景 |
| Domain relevance | 24 | 25 | 直接针对工业推荐 LLM 落地，快手广告平台与达人内容生态密切相关 |
| **Total** | **84** | **100** | |

---

## 相关链接

- arXiv: https://arxiv.org/abs/2606.03866
- 代码复现: `code/Taiji/`
