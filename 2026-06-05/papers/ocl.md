# Organizational Control Layer: Governance Infrastructure at the Execution Boundary of LLM Agent Systems

## 基本信息 / Basic Info

| Field | Value |
|-------|-------|
| Title | Organizational Control Layer: Governance Infrastructure at the Execution Boundary of LLM Agent Systems |
| arXiv | [2606.04306](https://arxiv.org/abs/2606.04306) |
| Submitted | June 3, 2026 |
| Authors | (Multiple — McGill, Purdue, UNSW, UCLA, NYU, Stevens, Aimaikj Research) |
| Affiliation | McGill / Purdue / UNSW / UCLA / NYU / Stevens Institute of Technology |
| Venue | arXiv preprint |
| Code | N/A |
| Domain Tag | LLM-governance · agent-safety · content-policy · execution-boundary |

---

## 方法概述 / Method Summary

**English:**  
LLM-based agents are increasingly deployed in workflows where model outputs directly trigger environment-changing actions (API calls, purchases, policy updates, etc.). The fundamental challenge—the **execution-boundary problem**—is that proposed actions must be evaluated and governed *before* they are committed, not after. The Organizational Control Layer (OCL) is a model-agnostic governance infrastructure placed at this boundary. It intercepts generated actions, applies declarative policy rules (expressed in natural language or structured logic), and either allows, modifies, or escalates the action before environment-facing execution. OCL is evaluated on adversarial buyer-seller negotiation environments across multiple frontier LLM backends.

**中文：**  
基于 LLM 的智能体越来越多地被部署在会直接触发真实操作（API 调用、下单、政策更新等）的工作流中。核心挑战——**执行边界问题**——在于必须在操作被提交前对其进行评估与治理，而非事后补救。组织控制层（OCL）是置于该边界处的模型无关治理基础设施：它拦截生成的操作，应用声明式策略规则（自然语言或结构化逻辑），在执行前进行放行、修改或升级人工审核。OCL 在多个前沿 LLM 后端的对抗性买卖双方谈判环境中进行了评估。

---

## 故事弧线 / Story Arc

> **传统方案的不足 →** 现有 LLM Agent 部署缺乏统一的执行前策略拦截机制，模型直接调用工具或 API，无法保证操作安全性和合规性。  
> **我们的方案 →** OCL 在 LLM 生成层与环境执行层之间插入可配置策略引擎，实现与模型无关的运行时治理。

---

## 关键指标 / Key Metrics

| Setting | Metric | OCL | Without OCL |
|---------|--------|-----|-------------|
| Adversarial negotiation (multi-LLM) | Unsafe Executions ↓ | **~0%** | **88%** |
| Adversarial negotiation (multi-LLM) | Valid Success ↑ | **96%** | 12% |

---

## 评分 / Scoring

| Dimension | Sub-score | Justification |
|-----------|-----------|---------------|
| Innovation (max 30) | 18 | Model-agnostic execution-boundary governance is a clean abstraction; conceptually solid |
| SOTA Delta (max 15) | 10 | 88%→~0% unsafe is dramatic, but only on adversarial toy environment |
| Experimental Quality (max 15) | 9 | Single task domain (negotiation); needs broader evaluation |
| Efficiency (max 10) | 7 | Policy intercept adds minimal latency; designed for lightweight operation |
| Generalization (max 5) | 4 | Multiple LLM backends; policy language is declarative and reusable |
| Domain Relevance (max 25) | 13 | Relevant to platform governance infrastructure, less directly to e-commerce content specifically |
| **Total** | **61** | |
