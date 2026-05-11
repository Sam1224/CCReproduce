# AI Co-Mathematician: Accelerating Mathematicians with Agentic AI

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| 标题 | AI Co-Mathematician: Accelerating Mathematicians with Agentic AI |
| 作者 | Daniel Zheng, Ingrid von Glehn, Yori Zwols, Iuliya Beloshapka, Lars Buesing, Daniel M. Roy, et al. (18 authors) |
| 机构 | Google DeepMind |
| arXiv | https://arxiv.org/abs/2605.06651 |
| 提交日期 | 2026-05-07 |
| 领域标签 | Agent · 多 agent 协作 · 数学推理 · 工作流 · DeepMind |
| 桶类别 | WEAK |
| 综合评分 | **67 / 100** |

---

## 方法概述 (中文)

**AI Co-Mathematician** 是 DeepMind 给数学研究者做的 agent 工作台：跟传统的"我问 LLM 一道题"不同，它把数学研究流程拆成几条平行管线——构思（ideation）、文献搜索、计算实验、定理证明、理论构建——每条管线由专门 agent 负责，研究者在中间做"导演"。

发现/来源：通过 **量子位** 公众号 mining 进入候选池（文章链接：https://www.qbitai.com/2026/05/414788.html ）。

---

## 故事线 (Story Arc)

> **现状不足：** 数学研究里 LLM 多以"问答助手"形式存在，无法持续推进开放问题。
>
> **我们的解法：** 把研究流程拆成异步 multi-agent workbench，每个 agent 负责一类任务，研究者保留 in-the-loop 决策权。

---

## 创新点分析

| 维度 | 描述 |
|------|------|
| 核心创新 | 数学研究 workflow 拆解为 agent 角色，强调"加速研究者"而非"替代研究者" |
| vs. 先前工作 | AlphaProof / AlphaGeometry 专攻单点定理；本工作覆盖完整研究流程 |
| 工程价值 | DeepMind 团队（18 作者）+ 实际帮研究者解决数论/群论 open problems |

---

## 关键指标

- FrontierMath Tier 4：**48%** (SOTA among all AI systems evaluated)
- 数学研究者实操使用案例（牛津教授 + 群论悬案）

---

## 评分分解

| 维度 | 得分 | 满分 | 说明 |
|------|------|------|------|
| 创新性 | 22 | 30 | Agentic workbench 范式新；但单点技术不算革命 |
| 实验 SOTA delta | 13 | 15 | FrontierMath Tier 4 48% 是 headline 数字 |
| 实验质量/消融 | 12 | 15 | 真实研究者使用案例 |
| 效率 | 5 | 10 | Agentic 推理成本高 |
| 泛化性 | 3 | 5 | 数学域；agentic 范式可迁移 |
| 领域相关性 | 12 | 25 | 数学非电商域；但 multi-agent workbench 模式可迁移到电商内容审核/达人助理 |
| **Total** | **67** | **100** | |

---

## 代码与数据

- 论文未公开仓库（Google DeepMind 内部研究系统）
- **本仓库提供完整 PyTorch 复现：** `code/AICoMath/`（架构骨架，非完整数学求解能力）
