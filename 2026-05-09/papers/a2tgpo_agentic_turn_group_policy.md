# A²TGPO: Agentic Turn-Group Policy Optimization with Adaptive Turn-level Clipping

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| 标题 | A²TGPO: Agentic Turn-Group Policy Optimization with Adaptive Turn-level Clipping |
| 作者 | Dingwei Chen, Zefang Zong, Zhipeng Ma, Leo Luo, Yang Li, Chengming Li, Peng Chen, Jie Jiang |
| 机构 | Tencent |
| arXiv | https://arxiv.org/abs/2605.06200 |
| 提交日期 | 2026-05-07 |
| 领域标签 | 多轮强化学习 · Agent · LLM 后训练 · Process Reward · 腾讯 |
| 桶类别 | WEAK |
| 综合评分 | **72 / 100** |

---

## 方法概述 (中文)

多轮 agentic LLM 训练（推荐对话、客服、达人助手等）面临一个老问题：每一轮的过程奖励该怎么分配。传统 GRPO 类方法对"早期、关键、信息量大"的轮和"末尾、冗余"的轮一视同仁，导致信用分配偏移。

**A²TGPO** 用 Information Gain 信号驱动三个改造：
1. **Turn-group normalization**：按交互深度分组归一化，让同一深度的轮可比；
2. **Variance-rescaled discounted accumulation**：用方差重缩放的折扣累积来稳定 advantage 量级；
3. **Adaptive turn-level clipping**：根据每轮信息量动态调整裁剪范围（高信息量轮许更大更新）。

整体作为对 GRPO 的可插拔增强，目标是把过程奖励"按需分配"到关键决策轮。

---

## 故事线 (Story Arc)

> **现状不足：** GRPO 类多轮 RL 把所有 turn 等权处理，过程信用分配偏移、关键轮被稀释。
>
> **我们的解法：** A²TGPO 用 Information Gain 做信号源，三个组件分别在归一化、累积、裁剪三层适配每轮的重要性。

---

## 创新点分析

| 维度 | 描述 |
|------|------|
| 核心创新 | Turn-group normalization + variance-rescaled discounted accumulation + adaptive clipping 的组合 |
| vs. 先前工作 | GRPO/REINFORCE 多轮版均匀对待 turns；本工作显式建模 turn 重要性 |
| 工程价值 | 腾讯出品；适配工业级多轮 agent 训练 |

---

## 关键指标

论文未公开 v1 详细数字（提交日期较新）。期望维度：agentic benchmark 上多轮 reward 收敛速度 + 最终 success rate。

---

## 评分分解

| 维度 | 得分 | 满分 | 说明 |
|------|------|------|------|
| 创新性 | 24 | 30 | 三个组件组合，针对过程信用分配老问题 |
| 实验 SOTA delta | 10 | 15 | v1 未给出 headline 数字 |
| 实验质量/消融 | 11 | 15 | 三个组件应需 ablation |
| 效率 | 6 | 10 | 不增推理代价；训练侧扩展 |
| 泛化性 | 4 | 5 | 适用所有 agentic RL 训练 |
| 领域相关性 | 17 | 25 | Tencent 多轮 agent 训练直接服务推荐/对话；电商场景可迁移 |
| **Total** | **72** | **100** | |

---

## 代码与数据

- 论文未公开仓库
- **本仓库提供完整 PyTorch 复现：** `code/A2TGPO/`
