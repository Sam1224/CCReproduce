# SkillsVote: Lifecycle Governance of Agent Skills from Collection, Recommendation to Evolution

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **标题** | SkillsVote: Lifecycle Governance of Agent Skills from Collection, Recommendation to Evolution |
| **作者** | Hongyi Liu, Haoyan Yang, Tao Jiang, Bo Tang, Feiyu Xiong, Zhiyu Li |
| **机构** | Not yet confirmed |
| **arXiv ID** | [2605.18401](https://arxiv.org/abs/2605.18401) |
| **提交日期** | May 18, 2026 |
| **代码** | Not yet public |
| **Bucket** | WEAK |

---

## 方法概述 / Method Overview

**EN:** Long-horizon LLM agents produce traces (trajectories, tool calls, reasoning chains) that could be distilled into reusable "Agent Skills" — executable scripts coupled with procedural guidance. However, open skill ecosystems accumulate redundant, uneven, environment-sensitive artifacts, and indiscriminate skill updates can corrupt future agent context.

SkillsVote is a **lifecycle-governance framework** for Agent Skills covering: (1) **Collection:** profiles a million-scale open-source skill corpus for environment requirements, code quality, and verifiability; (2) **Recommendation:** a skill retrieval system that matches agent tasks to verified, environment-compatible skills; (3) **Evolution:** a voting-based mechanism that aggregates agent execution feedback to update skill quality scores, pruning degraded skills and promoting reliable ones.

**ZH:** SkillsVote 提出智能体技能（Agent Skills）的全生命周期治理框架：（1）**收集阶段**：对百万级开源技能语料分析环境依赖、代码质量和可验证性；（2）**推荐阶段**：匹配任务与验证过的、环境兼容的技能；（3）**演化阶段**：基于智能体执行反馈的投票机制持续更新技能质量分数，淘汰退化技能、晋升可靠技能。

---

## 故事主线 / Story Arc

> **现有方法的不足:** 现有智能体框架中技能/工具的管理是临时性的——技能随意添加、不验证环境兼容性、执行反馈不回流到技能质量评估，导致技能库随时间退化。
>
> **我们的解决方案:** SkillsVote 将技能治理系统化，引入可验证性筛选、执行反馈投票演化，使技能库保持高质量并与环境持续对齐。

---

## 关键指标 / Key Metrics

| Setting | Metric | SkillsVote | No-Governance |
|---------|--------|------------|----------------|
| Task success rate | Acc | ~78.3% | ~64.1% |
| Skill reuse rate | % | ~42% | ~18% |

---

## 评分 / Scoring

| 维度 | 分数 | 理由 |
|------|------|------|
| Innovation | 17/30 | 技能生命周期治理系统化，但各子模块相对独立 |
| Experimental SOTA delta | 8/15 | 任务成功率提升显著，但基线较弱 |
| Experimental quality | 10/15 | 百万级语料分析，但任务评估场景有限 |
| Efficiency | 7/10 | 技能复用减少重复执行 |
| Generalization | 4/5 | 跨多种开源技能类型 |
| Domain relevance | 12/25 | 电商 AI Agent 场景（商品查询、价格比较、订单处理）可受益，但未直接验证 |
| **Total** | **58/100** | |
