# Code as Agent Harness

## 基本信息 / Basic Info

| 字段 | 内容 |
|---|---|
| **Title** | Code as Agent Harness: Toward Executable, Verifiable, and Stateful Agent Systems |
| **Authors** | 43 authors from various institutions |
| **Affiliation** | Multiple institutions (details in full paper) |
| **arXiv** | [2605.18747](https://arxiv.org/abs/2605.18747) |
| **Submitted** | 18 May 2026 |
| **Domain** | LLM Agents · Code Generation · Agentic Systems |
| **Bucket** | WEAK |
| **Code** | GitHub: https://github.com/YennNing/Awesome-Code-as-Agent-Harness-Papers |

---

## 分数 / Score

| 维度 | 分数 | 满分 |
|---|---|---|
| Innovation | 15 | 30 |
| Experimental SOTA delta | 5 | 15 |
| Experimental quality / ablations | 5 | 15 |
| Efficiency | 5 | 10 |
| Generalization | 5 | 5 |
| Domain relevance (ecom + governance) | 7 | 25 |
| **Total** | **42** | **100** |

**Justification**: Survey paper providing a useful taxonomy of code-as-harness for LLM agents. Innovation is limited (survey, not novel method). Domain relevance is marginal — code agents could be applied to e-commerce automation (pricing, inventory, content generation pipelines) but no direct application is demonstrated. Strong generalization score as a survey.

---

## 方法概述 / Method Summary

LLM 在代码生成方面表现出强大能力，但在新兴智能体系统（Agentic Systems）中，代码不再只是输出目标——它正在成为智能体推理、执行、状态管理的运行基础。

**"代码即智能体线束 (Code as Agent Harness)"** 的核心论点：
代码将模型输出转化为可执行、可检验、有状态的结构：
- 代码让**推理变得可执行**（CoT → 可运行的验证代码）
- 代码让**行动变得可编程**（工具调用 → 程序化接口）
- 代码让**环境状态变得可检验**（中间结果可验证）

**调查框架 — 三层结构**:

1. **线束接口层 (Harness Interface)**  
   代码如何连接智能体与推理、行动、环境建模。

2. **线束机制层 (Harness Mechanisms)**  
   规划（Planning）、记忆（Memory）、工具使用（Tool Use），以及反馈驱动的控制与优化。

3. **线束可靠性层 (Harness Reliability)**  
   如何使代码智能体在长序列执行中保持可靠和适应性。

**Story Arc**: LLM 代码生成已成熟，但其在智能体中的角色从"生成目标"到"执行基础"的转变缺乏系统梳理 → 本综述提供统一视角和研究路线图。

---

## 创新性分析 / Innovation Analysis

这是一篇综述/立场文章，创新性体现在概念框架的提炼而非算法创新：
- 首次从"线束"（Harness）视角统一了代码在 LLM 智能体中的多种角色；
- 提供了详尽的相关工作分类（43 位共同作者的大范围调研）；
- 识别了当前代码智能体的开放挑战（长序列稳定性、验证完整性、工具接口规范化）。

---

## 与电商内容生态的关联

- **电商自动化智能体**: 代码智能体可用于自动化内容生成（商品描述撰写、营销文案优化）、价格监控、库存管理等。
- **内容治理流水线**: 基于代码的智能体可构建可验证的内容审核流水线（规则 → 代码 → 可追溯的审核决策）。
- **数据处理自动化**: 数据清洗、标注的自动化可借助代码智能体实现。

---

## 论文链接

- arXiv: https://arxiv.org/abs/2605.18747
- GitHub 论文列表: https://github.com/YennNing/Awesome-Code-as-Agent-Harness-Papers
