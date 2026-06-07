# OCL: Organizational Control Layer — Governance Infrastructure at the Execution Boundary of LLM Agent Systems

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | Organizational Control Layer: Governance Infrastructure at the Execution Boundary of LLM Agent Systems |
| **Authors** | Tianyu Shi, Yang Mo, Yiou Liu, Zhuonan Hao, Yin Wang, Wenzhuo Hu, Nan Yu, Meng Zhou, Jiangbo Yu |
| **Affiliation** | McGill University, Purdue University, UNSW, UCLA, NYU, Stevens Institute of Technology, Aimaikj Research |
| **arXiv** | [2606.04306](https://arxiv.org/abs/2606.04306) |
| **Submitted** | ~June 3, 2026 (indexed June 5–6, 2026 GMT+8) |
| **Code** | Not reproduced (score 70 < 80 threshold) |
| **Bucket** | STRONG |

---

## 故事弧线 / Story Arc

> **"LLM Agent的输出可以直接触发高风险的执行操作，现有系统缺乏在执行边界的治理机制，导致88%的操作违规执行 → 我们设计OCL（组织控制层），在生成与执行之间插入与模型无关的策略执行层，将违规执行率降至接近零。"**

As LLM-based agents are deployed in workflows where generated text directly triggers real actions (purchases, API calls, financial transactions), the absence of a governance layer between language generation and environment-facing execution creates serious safety risks. OCL introduces a model-agnostic middleware layer that intercepts every proposed action, applies organizational policy constraints, and either approves execution, modifies the action, or escalates for human review.

---

## 方法 / Method

**OCL Architecture**:

```
LLM Agent (Proposal Generator)
        │ proposed action
        ▼
   ┌─────────────────────────┐
   │   Organizational        │
   │   Control Layer (OCL)   │
   │                         │
   │  ① Policy Checker       │ ← checks action against org policies
   │  ② Constraint Enforcer  │ ← clamps values to acceptable ranges
   │  ③ Escalation Engine    │ ← routes violations to human review
   │  ④ Deterministic        │
   │     Replanner           │ ← generates compliant alternative action
   └─────────────────────────┘
        │ approved/modified action
        ▼
  Environment Execution
```

**Key Design Principles**:
1. **Separation of concerns**: LLM proposes, OCL governs, environment executes
2. **Model-agnostic**: no modification to the underlying LLM
3. **Deterministic enforcement**: constraint violations trigger replanning, not re-prompting
4. **Adversarial robustness**: tested against prompt injection and adversarial buyer strategies

**Evaluation Setting**:
- Multi-agent buyer-seller negotiation adapted from AgenticPay benchmark
- Tests: unsafe price proposals, adversarial manipulation, policy compliance

---

## 创新性 / Innovation

| Aspect | Prior Work | OCL |
|--------|------------|-----|
| Governance location | Prompt-level (system prompts) | Execution boundary (intercept layer) |
| LLM modification required | Yes (in-context constraints) | No (model-agnostic middleware) |
| Enforcement | Soft (LLM may ignore) | Hard (deterministic constraint clamp) |
| Adversarial robustness | Limited | 60–94% intercept rate across backends |

**Novelty assessment**: The "execution boundary" framing is the key contribution — moving safety enforcement out of the LLM context and into the execution environment. This architectural separation is well-motivated for deployment scenarios. The evaluation on adversarial multi-agent negotiation is realistic and relevant.

---

## 关键指标 / Key Metrics

| Setting | Metric | OCL | Baseline |
|---------|--------|-----|---------|
| AgenticPay (GPT-5.4) | Unsafe execution rate | **~0%** | 88% |
| AgenticPay (GPT-5.4) | Valid success rate | **96%** | 12% |
| Adversarial intercept (GPT-5.4) | Intercept rate | **94%** | — |
| Adversarial intercept (Gemini-3.1) | Intercept rate | **82%** | — |
| Adversarial intercept (Qwen-3.5) | Intercept rate | **60%** | — |

---

## 评分 / Scoring

| Dimension | Max | Score | Justification |
|-----------|-----|-------|---------------|
| Innovation | 30 | 21 | Execution-boundary governance framing is novel and practically important; not deeply technically complex |
| Experimental SOTA delta | 15 | 11 | 88%→~0% unsafe executions is compelling; tested across 3 LLM backends |
| Experimental quality / ablations | 15 | 10 | Multi-backend, adversarial evaluation; benchmark is specialized (negotiation) |
| Efficiency | 10 | 7 | Deterministic replanning reduces wasted LLM turns; latency overhead from OCL intercept layer not fully analyzed |
| Generalization | 5 | 4 | Tested on 3 different LLM backends; task domain (negotiation) relatively narrow |
| Domain relevance (ecom + governance) | 25 | 17 | Agent governance broadly applicable to e-commerce agent systems; not directly e-commerce deployment |
| **Total** | **100** | **70** | Strong governance paper; relevant to building safe e-commerce agent systems; below code reproduction threshold |
