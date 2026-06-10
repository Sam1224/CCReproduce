# A General Framework for Multimodal LLM-Based Multimedia Understanding in Large-Scale Recommendation Systems

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | A General Framework for Multimodal LLM-Based Multimedia Understanding in Large-Scale Recommendation Systems |
| **Authors** | (Researchers from Meta Platforms) |
| **Affiliations** | Meta Platforms |
| **arXiv** | [2605.09338](https://arxiv.org/abs/2605.09338) |
| **Submitted** | 2026-05-10 |
| **Keywords** | MLLM、多模态理解、推荐系统、大规模部署、内容理解 |
| **Bucket** | STRONG |

---

## 方法概述 / Method Summary

工业级推荐系统需要深度理解多媒体内容（图像、视频、文本），但由于延迟约束和规模限制，直接在推荐管道中部署多模态大模型（MLLM）面临挑战。本文来自 Meta Platforms，提出一个**通用 MLLM 驱动的多媒体理解框架**：

1. **三部架构（Tripartite Architecture）**：
   - **内容解读层（Content Interpretation）**：基于 LLaMA2 的 MLLM 对多媒体内容生成描述性 caption；
   - **表征提取层（Representation Extraction）**：将生成的 caption tokenize 为离散类目特征；
   - **系统集成层（Pipeline Integration）**：将提取的特征无缝注入现有推荐管道（特征工程层面集成，而非端到端替换）。

2. **工程导向设计**：延迟约束下，MLLM 在线下运行生成语义标签，推理时只使用轻量离散特征，零延迟增量。

3. **规模化实践**：在 Meta 内部的工业级推荐系统上验证，覆盖图像、视频等多种内容类型。

---

## 故事弧 / Story Arc

> **现状不足** → **提出方案**

传统推荐系统依赖人工标注的分类标签和简单视觉特征（如 ResNet 嵌入），无法充分挖掘多媒体内容的高维语义。将 MLLM 直接嵌入推理管道延迟过高。

本框架通过"离线 MLLM 理解 + 在线离散特征消费"的解耦设计，将 MLLM 的语义理解能力引入工业推荐，同时满足延迟约束。

---

## 创新性分析 / Innovation

| 维度 | 分析 |
|------|------|
| vs. 端到端 MLLM 推荐 | 解耦离线-在线，满足工业延迟要求 |
| vs. 传统内容标注 | 利用 MLLM 替代人工标注，提升语义丰富度 |
| vs. 视觉特征工程 | caption 转类目特征更紧凑，更易于现有系统集成 |

**可行性评估**：高。离线处理 + 在线使用是成熟的工业模式。

---

## 关键指标 / Key Metrics

| 数据集/场景 | 指标 | MLLM Framework | Baseline |
|-------------|------|----------------|----------|
| Meta 工业推荐（在线） | CTR | +提升 | 传统特征 |
| Meta 工业推荐（在线） | 用户参与度 | +提升 | — |

---

## 评分明细 / Score Breakdown

| 维度 | 得分 | 满分 | 说明 |
|------|------|------|------|
| 方法创新性 | 18 | 30 | 工程框架贡献为主，无新型模型结构，但工业落地思路清晰 |
| 实验指标 SOTA | 11 | 15 | Meta 线上验证，但数值不公开 |
| 实验质量/消融 | 10 | 15 | 有框架各组件贡献分析 |
| 方法效率 | 9 | 10 | 离线处理设计完全不增加推理延迟 |
| 方法泛化性 | 4 | 5 | 通用框架，理论上适用多类推荐场景 |
| 论文相关性 | 20 | 25 | MLLM 辅助推荐系统内容理解，中强相关 |
| **Total** | **72** | **100** | Meta 工业实践，框架通用性强，对内容生态理解有直接参考价值 |
