# RUBAS: Rubric-Based Reinforcement Learning for Agent Safety

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **arXiv** | https://arxiv.org/abs/2606.04051 |
| **提交日期** | 2026-06-02（公告窗口 2026-06-03~04 GMT+8） |
| **作者** | 未从 snippet 中获取（原文待查） |
| **机构** | 未公开披露 |
| **领域标签** | `Agent Safety` `Reinforcement Learning` `Tool Use` `Rubric-Based Reward` `LLM Alignment` |
| **桶位** | WEAK |

---

## 方法概述

LLM 进化为工具使用 agent 后，带来一类新的安全挑战：**工具调用安全性**（tool-use safety）。现有对齐方法（RLHF/DPO/Constitutional AI）使用粗粒度拒绝信号（refuse or not）或静态监督，难以在多样 agentic 风险下同时保证安全与有用性。

RUBAS 将 agent 行为分解为四个维度：
1. **工具调用安全性（Tool-Use Safety）**：工具本身是否被允许调用？
2. **参数安全性（Argument Safety）**：工具调用的参数是否合规（如 SQL 注入、路径穿越等）？
3. **响应安全性（Response Safety）**：工具返回结果后 LLM 的回复是否安全？
4. **有用性（Helpfulness）**：在保证安全的前提下，任务是否被有效完成？

对应四个维度设计结构化 Rubric，生成细粒度、可解释的奖励信号，在完整 agent trajectory 上执行 RL 优化（非 step-level），实现安全工具调用与任务完成的联合优化。

---

## 故事弧线 / Story Arc

**Agent 工具调用产生执行级安全风险，现有粗粒度对齐信号无法精确指导** → RUBAS 把 agent 行为分解为四维度，设计细粒度 Rubric 作为结构化奖励 → RL 在完整 trajectory 上优化 → 安全性提升，工具 hallucination 减少，有用性维持竞争力。

---

## 创新性分析

- **四维度分解**是对 agent safety RL 的有价值结构化，比粗粒度 refusal 信号更精确。
- Rubric-based reward 与基于人类反馈的 reward model 相比，具有更好的可解释性和规则可维护性。
- 适合电商平台上"商家工作台 agent"或"客服 agent"等需要精细安全控制的场景。
- **局限**：Rubric 设计依赖领域专家，跨场景迁移需重新设计；RL 训练稳定性未详述。

---

## 关键指标

| 评测 | 指标 | RUBAS vs. Baseline |
|------|------|-------------------|
| 多个 agent safety benchmarks | 安全率 | 优于标准对齐 baseline |
| 工具调用 hallucination | Hallucination 率 | 显著下降 |
| 任务完成率 | Utility | 竞争力保持 |

- 具体数字未从 snippet 获取。

---

## 评分 (满分 100)

| 维度 | 分 / 满分 | 理由 |
|------|----------|------|
| 创新性 | 20 / 30 | 四维度 Rubric 设计新颖；RL 框架常规 |
| 实验指标 | 9 / 15 | 多 benchmark 正向；无具体数字 |
| 实验质量 | 10 / 15 | 多 benchmark、多模型实验 |
| 效率 | 7 / 10 | 无需修改 LLM 架构 |
| 泛化性 | 3 / 5 | 多模型验证 |
| 相关性 | 13 / 25 | 内容治理 agent、商家 agent 安全性间接相关 |
| **Total** | **62** |
