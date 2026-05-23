# SkillsVote: Lifecycle Governance of Agent Skills from Collection, Recommendation to Evolution

## 基本信息 / Basic Info

| 字段 | 值 |
|------|-----|
| **Title** | SkillsVote: Lifecycle Governance of Agent Skills from Collection, Recommendation to Evolution |
| **arXiv** | https://arxiv.org/abs/2605.18401 |
| **Authors** | Hongyi Liu, Haoyan Yang, Tao Jiang, Bo Tang, Feiyu Xiong, Zhiyu Li |
| **Affiliation** | (Not retrieved) |
| **Submitted** | May 18–19, 2026 |
| **Domain Tags** | `agent-skills` `lifecycle-governance` `knowledge-management` `RAG` `skill-recommendation` |
| **Code** | Not provided |
| **Bucket** | WEAK |
| **Total** | **55** |

---

## 方法概述 / Method Overview

**中文：**
**SkillsVote** 是一个 LLM Agent 技能的全生命周期治理框架，解决开放式 agent skill 生态中的冗余、质量不均和环境敏感性问题。框架将 agent skills 定义为一种"经验模式（experience schema）"，耦合可执行脚本和不可执行的程序性指引。SkillsVote 对百万级开源语料库进行画像（环境需求、质量、可验证性），为可验证的 skill 合成任务，并在执行前通过 agentic 库搜索为 agent 提供结构化技能上下文。目标是构建高质量、可治理的 agent skill 知识库，防止低质量更新污染后续 context。

**English:**
SkillsVote is a lifecycle governance framework for LLM agent skills, addressing redundancy, uneven quality, and environment-sensitivity in open skill ecosystems. Agent Skills are defined as an experience schema coupling executable scripts with non-executable procedural guidance. SkillsVote profiles a million-scale open-source corpus (environment requirements, quality, verifiability), synthesizes tasks for verifiable skills, and performs agentic library search before execution to expose structured skill context. The goal is to prevent indiscriminate updates from polluting the skill knowledge base.

---

## 故事线 / Story Arc

LLM agent 的执行轨迹可转化为可复用的技能（skills）→  
开放 skill 生态中存在大量冗余、低质量或环境不兼容的 skill →  
直接使用会污染 agent 的 context →  
SkillsVote 通过质量评估、可验证性过滤和执行前库搜索实现技能的全生命周期治理 →  
百万级语料库画像确保技能库的持续演进不降级。

---

## 创新性分析 / Innovation

- **生命周期视角**：将 skill 管理从"收集"延展到"推荐"和"演进"，全流程治理是有意义的贡献。
- **可验证性过滤**：专门关注技能的可验证性（verifiability），对建立可信 skill 知识库至关重要。
- **局限**：框架较宏观，与电商具体场景关联需要额外映射；agent skill 治理更多是 AI 基础设施问题。

---

## 关键指标 / Key Metrics

| Setting | Metric | Value |
|---------|--------|-------|
| Open-source corpus | Scale | Million-scale profiles |
| Skill quality | Verifiability filtering | Demonstrated reduction in low-quality skills |

---

## 评分 / Scoring

| 维度 | 分 / 满分 | 理由 |
|------|----------|------|
| 创新性 | 17 / 30 | 全生命周期视角有价值；但技术细节未充分披露 |
| 实验指标 | 7 / 15 | 百万级语料库，但量化指标有限 |
| 实验质量 | 8 / 15 | 系统性框架，但具体实验细节未完全获取 |
| 效率 | 6 / 10 | Agentic 库搜索有一定开销 |
| 泛化性 | 3 / 5 | 通用 agent 系统，可适配电商 agent |
| 相关性 | 14 / 25 | 间接相关：电商 AI agent（搜索/推荐/客服）的技能治理是重要基础设施问题 |
| **Total** | **55** | |
