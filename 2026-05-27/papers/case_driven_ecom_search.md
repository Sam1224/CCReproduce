# A Case-Driven Multi-Agent Framework for E-Commerce Search Relevance

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | A Case-Driven Multi-Agent Framework for E-Commerce Search Relevance |
| **Authors** | ByteDance Global E-Commerce Search Relevance Team (corresponding: wangfeiyi@bytedance.com) |
| **Affiliations** | ByteDance |
| **arXiv** | [2605.05991](https://arxiv.org/abs/2605.05991) |
| **Submitted** | 2026-05-06 |
| **Bucket** | STRONG |
| **Total** | **76 / 100** |

---

## 方法概述 / Method Overview

### EN
Industrial e-commerce search relevance pipelines suffer from "bad cases" — product-query pairs with poor relevance signals — that degrade user experience. Manual identification and resolution of these cases is costly, slow, and inconsistent. This paper proposes a **case-driven multi-agent framework** with three specialized agents: (1) an **Annotator Agent** for multi-turn relevance annotation that minimizes hallucination and variance; (2) an **Optimizer Agent** that autonomously analyzes and resolves bad cases; and (3) a **User Agent** that surfaces bad cases through conversational interaction. These are united by a **Global Memory** module that reduces information asymmetry across agents. The system also integrates a **Deep Search Agent** targeting underestimation failures and an **agent-based chatbot** for human-agent collaboration. A unified retrieval-and-ranking relevance model enables efficient training, while an instruction-following relevance model handles real-time resolution.

### ZH
电商搜索相关性系统中的"Bad Case"（低质量相关性标注）会严重影响用户体验，人工发现和修复成本高、速度慢、一致性差。本文提出**案例驱动多智能体框架**，包含三个专用 Agent：(1) **标注 Agent**：多轮标注，减少幻觉和方差；(2) **优化 Agent**：自主分析并解决 Bad Case；(3) **用户 Agent**：通过对话挖掘 Bad Case。三者由**全局记忆**模块协同，降低信息不对称。系统还集成**深度搜索 Agent**（针对漏召问题）和基于 Agent 的人机协作聊天机器人。统一的检索-排序相关性模型支持高效训练，指令跟随相关性模型支持实时修复。

---

## 故事弧 / Story Arc

> **"电商搜索 Bad Case 人工识别成本高、迭代慢"** → 设计三 Agent 自动发现-标注-修复闭环，配合全局记忆和 Deep Search Agent，持续优化搜索相关性。

---

## 创新性分析 / Innovation Analysis

1. **闭环多 Agent 架构**：将标注、优化、发现三个环节 Agent 化并形成自动迭代闭环，在工业级系统中实属少见。
2. **Global Memory 设计**：跨 Agent 全局记忆减少重复分析和信息孤岛，类似于软件工程中的共享状态管理。
3. **Bad Case 主动发现**：传统方法被动等待 Bad Case 上报，User Agent 主动通过对话挖掘，更早介入。
4. **Harness Engineering 范式**：生产级部署约束下的 Agent 系统设计范式，可迁移至其他工业场景。

可行性：ByteDance 有大规模电商搜索系统作为验证平台，技术路线成熟。

---

## 关键指标 / Key Metrics

| Metric | Result |
|--------|--------|
| Annotation accuracy (human eval) | 显著提升（具体数值未披露） |
| Bad case resolution timeliness | 更快（具体数值未披露） |
| Generalizability of resolution | 覆盖更广泛的 case 类型 |

*注：论文以人工评测为主，未提供精确数值*

---

## 评分明细 / Scoring Breakdown

| 维度 | 分值 | 得分 | 说明 |
|------|------|------|------|
| Innovation | 30 | 22 | 工业级多 Agent 闭环，Global Memory 设计新颖 |
| Experimental SOTA delta | 15 | 10 | 人工评测定性结论，缺乏量化 SOTA 对比 |
| Experimental quality / ablations | 15 | 10 | 生产部署验证有说服力，但数字披露不足 |
| Efficiency | 10 | 7 | 实时分辨率模型兼顾效率 |
| Generalization | 5 | 3 | 搜索相关性领域专用 |
| Domain relevance | 25 | 24 | ByteDance 电商搜索，直接对应电商内容质量治理 |
| **Total** | **100** | **76** | |

---

## 电商治理价值 / E-commerce Governance Value

- **商品相关性标注**：Annotator Agent 可直接用于大规模商品-查询相关性打标，替代昂贵人工标注
- **达人内容审核**：框架可扩展为内容违规 Bad Case 发现-修复闭环
- **搜索质量治理**：电商平台标品/非标品查询匹配质量持续优化
