# Multi-Agent Reinforcement Learning from Delayed Marketplace Feedback for Objective-Weight Adaptation in Three-Sided Dispatch

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | Multi-Agent Reinforcement Learning from Delayed Marketplace Feedback for Objective-Weight Adaptation in Three-Sided Dispatch |
| **ArXiv ID** | [2606.13604](https://arxiv.org/abs/2606.13604) |
| **Authors** | Haochen Wu, Yi Hou, Ryan Xie |
| **Affiliation** | DoorDash |
| **Submitted** | 2026-06-11 |
| **Source** | HuggingFace June 13 daily listing |
| **Bucket** | WEAK — 平台调度治理、Offline RL、延迟反馈 |

---

## 方法概述 / Method Overview

**故事弧线：** 三方市场（商家/骑手/用户）的调度系统用固定权重启发式，面对局部拥堵/供需变化时脆弱；直接用 RL 替代组合优化器风险高、学习难度大。→ DoorDash 采用"窄接口"方案：RL 仅选择离散 objective-weight multiplier 调节优化器目标（速度 vs 拼单效率），保留原优化器安全护栏，用延迟市场反馈做 offline RL。

**技术要点：**
- MARL + 集中式 value function + 去中心化执行（store-level agents）；
- Double Q + 保守正则缓解 OOD 价值过估（offline RL 标准挑战）；
- 延迟奖励信号来自区域级市场结果（取餐时间/拼单率），体现网络效应；
- Switchback 实验设计（约 4,000 区域，2 小时切换）严格控制混淆。

---

## 关键指标 / Key Metrics

| 指标 | 全天 | p值 | Dinner 峰值 |
|------|------|-----|------------|
| CAT (Click-to-Assign Time) | **-1.261s** | 0.019 | -1.289s |
| CWT (Consumer Wait Time) | **-0.856s** | 0.004 | -1.030s |
| % batched (↑) | **+0.495pp** | <0.001 | +0.600pp |
| %20-min late | -0.012pp | ns | -0.037pp (p=0.04) |

---

## 评分 / Scoring

| 维度 | 得分 | 满分 | 说明 |
|------|------|------|------|
| 方法创新性 | 20 | 30 | 窄接口 + offline MARL 组合有工程创意 |
| 实验指标 | 10 | 15 | 线上 switchback，多指标显著 |
| 实验质量 | 12 | 15 | 大规模 AB，统计严谨，区域级 power 分析 |
| 方法效率 | 8 | 10 | 保留原优化器，RL 开销仅在 multiplier 选择 |
| 方法泛化性 | 2 | 5 | 专为三方配送市场设计 |
| 领域相关性 | 15 | 25 | 平台调度/效率治理有借鉴价值，但非内容理解/达人治理主线 |
| **Total** | **67** | **100** | |

---

## Story Arc

> **现状不足：** 固定权重调度规则在动态供需下脆弱；直接 RL 替换优化器风险高。  
> **解法：** RL 限制为 objective-weight 选择（窄接口） → 用区域延迟反馈构造奖励 → 保守正则 offline RL → switchback 验证速度与拼单同步改善。
