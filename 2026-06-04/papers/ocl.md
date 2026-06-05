# OCL: Organizational Control Layer — Governance Infrastructure at the Execution Boundary of LLM Agent Systems

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **arXiv** | https://arxiv.org/abs/2606.04306 |
| **提交日期** | 2026-06-03（公告窗口 2026-06-04 GMT+8） |
| **作者** | Tianyu Shi, Yang Mo, Yiou Liu, Zhuonan Hao, Yin Wang, Wenzhuo Hu, Nan Yu, Meng Zhou, Jiangbo Yu |
| **机构** | McGill University, Purdue University, UNSW Sydney, UCLA, NYU, Stevens Institute of Technology, Aimaikj Research |
| **领域标签** | `Agent Governance` `LLM Safety` `E-commerce Agents` `Policy Enforcement` `执行边界` |
| **桶位** | STRONG |
| **代码** | `code/OCL/` (本仓库复现) |

---

## 方法概述

LLM-based agent 系统越来越广泛地部署在能直接触发状态变化动作的工作流中（如电商平台的买卖双方谈判、自动下单、价格修改等），由此产生了**执行边界问题（execution-boundary problem）**：LLM 生成的提案需要在被实际执行前受到治理，而现有方案只在 LLM 输入/输出层做 guardrail，完全不感知"动作执行"这一关键环节。

OCL（Organizational Control Layer）是一个**模型无关的治理基础设施**：它在 LLM agent 生成动作提案之后、环境实际执行之前，插入一个拦截层，通过**策略强制（policy enforcement）+ 升级机制（escalation）**，对提案进行合规检查与过滤，而无需修改底层 LLM。

### 核心设计

```
LLM Generator → [Action Proposal] → OCL Interceptor → [Policy Check]
                                             ↓              ↓
                                     Approved Action  Blocked / Escalate
                                             ↓
                                     Environment Execution
```

1. **提案-执行分离**：LLM 负责提案（proposal），OCL 负责 execution gate。两者解耦，LLM 可以替换。
2. **策略强制层**：可插拔的策略规则（可以是符号规则或小型分类器），对提案的动作类型、参数范围、上下文合规性进行检查。
3. **升级机制**：当策略层置信度低或情境模糊时，将决策升级给人工审核或更保守的 fallback 策略，避免"二元硬拦截"造成的过度保守。
4. **无需修改 LLM**：OCL 是纯外挂层，与 GPT-4o、Claude、Gemini 等任意前沿模型兼容。

---

## 故事弧线 / Story Arc

**现有 LLM 部署只做 input/output guardrail，不治理"动作执行"** → 在电商谈判、批量操作等经济后果明显的场景，LLM 提案若直接执行，不合规率高达 88%，合法成功率仅 12% → **OCL 在执行边界插入策略拦截层，做提案-执行的组织级分离** → 不合规执行降至接近 0，合法成功率提升到 96%。

---

## 创新性分析

- **执行边界这一抽象是新颖贡献**：以往 LLM safety 工作关注输出文本的 refusal 或 jailbreak，OCL 把问题重构为"组织级别的动作治理"，更接近工业软件系统中的 authorization layer 概念。
- **提案-执行解耦架构**：与 Constitutional AI、RLHF 等修改 LLM 本身的路线正交，也与 tool-use guardrail（只检查工具调用参数）不同——OCL 在更高的抽象层（整个 action plan）做检查。
- **可插拔策略**：支持符号规则、ML 分类器、人工审核，避免一刀切。
- 与 ReAct / AgentPay / AutoGPT 等框架可直接集成，不依赖特定 agent 框架。
- **局限**：仅在买卖双方谈判单一 adversarial 环境中评估；策略层设计依赖专家规则，自动策略生成尚未覆盖。

---

## 关键指标

| 数据集/环境 | 指标 | OCL | Baseline (无 OCL) |
|------------|------|-----|------------------|
| AgenticPay 对抗谈判（多个前沿 LLM backends） | 不合规执行率 | **~0%** | 88% |
| AgenticPay 对抗谈判 | 合法任务成功率 | **96%** | 12% |

- 评测采用多个前沿 LLM 后端（GPT-4o 系列、Claude 系列等）；
- 对抗环境中买方 agent 试图诱使卖方 agent 执行非合规价格修改或条款变更。

---

## 评分 (满分 100)

| 维度 | 分 / 满分 | 理由 |
|------|----------|------|
| 创新性 | 24 / 30 | 执行边界抽象 + 组织级治理架构是新颖视角；策略层设计相对保守 |
| 实验指标 | 13 / 15 | 88%→~0% 不合规；12%→96% 成功率，幅度极大 |
| 实验质量 | 10 / 15 | 多 LLM backend；单一对抗环境，ablation 细节有限 |
| 效率 | 8 / 10 | 无需修改 LLM，拦截层轻量 |
| 泛化性 | 4 / 5 | 多前沿 LLM backend，但仅一个 domain |
| 相关性 | 21 / 25 | 电商平台达人 agent / 商家 agent 的执行合规治理直接适用 |
| **Total** | **80** | ✅ ≥80，进入复现 `code/OCL/` |

---

## 复现状态

- `code/OCL/`: PyTorch 实现的 OCL 框架原型，含 PolicyLayer（符号规则 + 小型 MLP 分类器）、EscalationHandler、AgentEnvironment 模拟、买卖双方 adversarial 测试。
