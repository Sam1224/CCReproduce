# Valley3: Scaling Omni Foundation Models for E-commerce

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| 标题 | Valley3: Scaling Omni Foundation Models for E-commerce |
| 作者 | Zeyu Chen, Guanghao Zhou, Qixiang Yin, Ziwang Zhao, Huanjin Yao, Pengjiu Xia, Min Yang, Cen Chen, Minghui Qiu |
| 机构 | 字节跳动 (ByteDance) — Valley 团队 |
| arXiv | https://arxiv.org/abs/2605.01278 |
| GitHub | https://github.com/bytedance/Valley |
| 提交日期 | 2026-05-02 (v1) |
| 领域标签 | 电商 MLLM · 全模态 · 短视频 · 直播 · 内容审核 · 商品理解 |
| 桶类别 | STRONG |
| 综合评分 | **76 / 100** |

---

## 方法概述 (中文)

电商场景涉及**多模态输入**（商品图片、短视频、直播流、商品描述文本、用户语音询问）以及**多样化任务**（商品理解、售后体验、搜索&推荐、直播内容分析、短视频理解、内容审核&治理），现有 VLM 主要面向图文对，无法覆盖音频模态，也缺乏电商垂直领域知识。

**Valley3** 是首个面向全球电商场景的**全模态（Omni）大语言模型**：
- **骨干架构：** 基于 Qwen3-VL（视觉语言）+ Qwen3-Omni（原生音频理解）扩展而来
- **四阶段预训练管线：**
  1. 音频理解对齐（Audio Understanding Alignment）
  2. 跨模态指令跟随（Cross-modal Instruction Following）
  3. 电商领域知识注入（E-commerce Domain Knowledge）
  4. 长上下文推理能力（Long-context Reasoning）
- **电商 Agent 能力：** 内置主动搜索工具调用（Search Tool Invocation），支持电商深度研究任务
- **自建基准（Omni E-commerce Benchmark）：** 覆盖 6 个核心任务：商品理解、售后体验、搜索&推荐、直播内容分析、短视频理解、**内容审核&治理**

---

## 故事线 (Story Arc)

> **现状不足：** 通用 VLM 缺乏音频模态支持，也没有电商领域特化知识；现有电商模型（如 Valley2）无法处理音视频混合场景（如直播、短视频），且不具备 Agent 工具调用能力。
>
> **我们的解法：** Valley3 通过四阶段持续预训练，将通用 Qwen3-VL 扩展为支持 Text+Image+Video+Audio 的全模态电商专家模型，并在 6 任务自建基准上持续超越强基线，同时保持通用域竞争力。

---

## 创新点分析

| 维度 | 描述 |
|------|------|
| 核心创新 | 首个全模态（文/图/视频/音频）电商专用 MLLM，覆盖直播音频理解 |
| vs. 先前工作 | Valley/Valley2 只支持图文/视频；通用 VLM（GPT-4V, Qwen-VL）缺乏电商垂直知识 |
| 基准创建 | Omni E-commerce Benchmark 填补了全模态电商评测空白 |
| Agent 能力 | 内置搜索工具调用，为电商深度研究场景提供 Agentic 支持 |
| 局限 | 四阶段训练细节和数据量未完全公开；音频理解能力的边界测试不足 |

---

## 关键指标

| 数据集 | 任务 | 结果 |
|--------|------|------|
| Omni E-commerce Benchmark | 商品理解 (Product Understanding) | SOTA（优于强基线）|
| Omni E-commerce Benchmark | 直播内容分析 (Livestream Content Analysis) | SOTA |
| Omni E-commerce Benchmark | 内容审核&治理 (Moderation & Governance) | SOTA |
| Omni E-commerce Benchmark | 搜索&推荐 (Search & Recommendation) | SOTA |
| 通用基准 (MMMU, etc.) | 通用多模态理解 | 竞争力保持（未退化）|

---

## 评分分解

| 维度 | 得分 | 满分 | 说明 |
|------|------|------|------|
| 创新性 (Innovation) | 21 | 30 | 全模态+电商垂直的组合创新，但技术路线属已知方法组合 |
| 实验 SOTA Delta | 10 | 15 | "持续优于强基线"但具体数值未在公开搜索结果中给出 |
| 实验质量/消融 | 9 | 15 | 6 任务基准覆盖全面，但四阶段消融细节未公开 |
| 效率 | 6 | 10 | 持续预训练相对高效，但全模态推理成本较高 |
| 泛化性 | 5 | 5 | 同时保持通用域竞争力 |
| 领域相关性 | 25 | 25 | 完全面向电商场景，含直播/短视频/内容审核六大任务 |
| **总分** | **76** | **100** | — |

---

## 代码与数据

- GitHub 仓库：https://github.com/bytedance/Valley（已有历史版本，Valley3 代码待发布）
- 模型权重：待公开
- 数据集（Omni E-commerce Benchmark）：内部，待开放
