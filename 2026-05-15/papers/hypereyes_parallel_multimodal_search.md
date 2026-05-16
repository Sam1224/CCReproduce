# HyperEyes: Dual-Grained Efficiency-Aware Reinforcement Learning for Parallel Multimodal Search Agents

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | HyperEyes: Dual-Grained Efficiency-Aware Reinforcement Learning for Parallel Multimodal Search Agents |
| **arXiv ID** | 2605.07177 |
| **Submitted** | 2026-05-07 |
| **Link** | https://arxiv.org/abs/2605.07177 |
| **Authors** | (DeepExperience team — from GitHub repo) |
| **Affiliation** | TBC |
| **Code** | https://github.com/DeepExperience/HyperEyes |
| **Venue** | arXiv preprint |
| **Bucket** | WEAK |

---

## 方法概述 / Method Overview

**问题（Story Arc）：** 现有多模态搜索 Agent 对独立子检索任务仍按序调用工具（one tool call per entity per round），导致冗余交互轮次爆炸；同时，现有 benchmark 只评估准确率而忽略推理效率，无法反映真实部署成本。

**HyperEyes 框架（三大创新）：**  
1. **并行多实体搜索**：将视觉 grounding + 检索融合为单一原子动作（atomic action），在一轮内并发搜索多个实体，减少不必要的顺序依赖。
2. **TRACE（Tool-use Reference-Adaptive Cost Efficiency）**：轨迹级奖励函数，基于参考工具调用数的单调收紧机制，训练过程中逐步压缩无效工具调用（macro-level）。
3. **On-Policy Distillation**：对失败 rollout 注入外部 teacher 的 token 级纠错信号，弥补稀疏奖励信号的信用分配缺陷（micro-level）。
4. **IMEB（Efficiency-Aware Benchmark）**：首个同时评估搜索能力和推理效率的人工标注 benchmark（300 个实例）。

---

## 创新性分析 / Innovation Analysis

| 维度 | 分析 |
|------|------|
| vs. 串行搜索 agent | 并行原子动作大幅减少 tool call 轮次，推理成本显著降低 |
| vs. 单纯准确率优化 | TRACE 将效率作为 first-class 训练目标，而非事后指标 |
| vs. 标准 GRPO | On-Policy Distillation 解决失败轨迹的学习信号稀疏问题 |
| IMEB 贡献 | 填补了多模态搜索效率评测的空白 |

---

## 关键指标 / Key Metrics

| Benchmark | Metric | HyperEyes-30B | Best Open-Source |
|-----------|--------|---------------|-----------------|
| 6 Search Benchmarks (avg) | Accuracy | +9.9% (relative) | Top open-source agent |
| 6 Search Benchmarks (avg) | Tool-call Rounds | 5.3× fewer | Top open-source agent |
| IMEB (joint) | Efficiency-Accuracy Score | state-of-the-art | — |

---

## 评分明细 / Scoring Breakdown

| 维度 | 满分 | 得分 | 说明 |
|------|------|------|------|
| Innovation | 30 | 21 | 并行原子动作 + TRACE 效率奖励 + IMEB 基准三合一 |
| SOTA Delta | 15 | 12 | +9.9% 准确率 + 5.3× 效率提升 |
| Experimental Quality | 15 | 11 | 6 benchmark + IMEB + ablation |
| Efficiency | 10 | 8 | 核心贡献之一：5.3× 工具调用减少 |
| Generalization | 5 | 3 | 多搜索任务类型 |
| Domain Relevance | 25 | 5 | 通用搜索 agent，非直接电商/治理，但底层技术可迁移 |
| **Total** | **100** | **60** | WEAK |

---

## 故事弧 / Story Arc

> "多模态搜索 agent 顺序工具调用导致轮次冗余，且现有评测忽视效率成本 → HyperEyes 以并行原子动作减少冗余，TRACE 奖励在训练中压缩工具调用，IMEB 基准重新定义效率评测，HyperEyes-30B 在准确率+效率双维超越最强开源 agent。"
