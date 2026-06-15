## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **标题** | Evoflux: Inference-Time Evolution of Executable Tool Workflows for Compact Agents |
| **作者** | Kushal Raj Bhandari, Ling Yue, Ching-Yun Ko, Dhaval Patel, Shaowu Pan, Pin-Yu Chen, Jianxi Gao |
| **机构** | Rensselaer Polytechnic Institute, IBM Research |
| **arXiv** | [2606.12674](https://arxiv.org/abs/2606.12674) |
| **代码** | [github.com/IBM/Evoflux](https://github.com/IBM/Evoflux) |
| **发布日期** | 2026-06-12 |
| **领域标签** | Agent、工具工作流、MCP、推理时搜索、小模型、可执行修复 |

---

## 方法概述

**Story Arc**：小模型工具 Agent 的失败点不在"会不会写 JSON"，而在于动态工具目录下生成的 workflow 在参数校验、依赖链传递与执行落地时频繁崩溃。仅靠少量 teacher traces SFT 只能学到成功样式，学不到修复行为。Evoflux 转换思路：把 compute 花在 test-time repair 上，核心为：

1. **Typed Workflow Graph 表示**：把工具调用序列形式化为有 schema 和依赖约束的有向无环图；
2. **结构化 Edits + 执行反馈**：在固定预算内通过 edit 操作（添加、替换、参数修正）生成候选修复；
3. **自适应强度控制 + 多样性剪枝**：避免退化到局部最优，提高修复成功率；
4. **执行可行性（Execution Feasibility）**作为唯一核心目标，而非 semantic similarity。

**Innovation vs Prior Work**：ReAct/Toolformer 等方案依赖 LLM 自生成修复，对小模型效果差；Best-of-N 等 tree search 方法缺乏 schema 约束导致无效搜索空间。Evoflux 引入 schema-grounded 进化搜索，并量化了小数据 SFT 的潜在负效应（performance degradation on live tool catalog）。

---

## 关键指标

| 数据集 | 指标 | Evoflux | SFT / SFT+DPO |
|--------|------|---------|--------------|
| MCP-Bench (28 servers, 250 tools) | Execution Feasibility | **17–24%** | 往往不升反降 (vs ~3% base) |
| MCP-Bench held-out tasks | Feasibility lift vs base | **+14–21pp** | SFT 有时降低可用性 |

---

## 评分

| 维度 | 得分 | 说明 |
|------|------|------|
| 方法创新性 | 23/30 | Workflow repair + execution grounding 是新视角，但进化搜索本身非首创 |
| 实验指标 | 12/15 | 可用性从 3% 到 17–24% 是明显提升，且 SFT 负效应对比有说服力 |
| 实验质量 | 12/15 | MCP-Bench 为真实工具服务评测，对比多基线 |
| 效率 | 7/10 | test-time search 增加推理开销，但避免重新训练 |
| 泛化性 | 4/5 | MCP-style tool use 框架通用性强 |
| 领域相关性 | 22/25 | 与电商/治理 Agent 多工具编排直接相关 |
| **Total** | **80/100** | 小模型 tool-use 可用性的关键突破，开源代码可直接对接 |

---

## 复现

官方代码：[github.com/IBM/Evoflux](https://github.com/IBM/Evoflux) — 已验证。  
本仓库路径：`2026-06-14/Evoflux/`（含 verify_repo.py）
