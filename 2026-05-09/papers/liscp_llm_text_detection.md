# Lightweight Stylistic Consistency Profiling: Robust Detection of LLM-Generated Textual Content for Multimedia Moderation

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| 标题 | Lightweight Stylistic Consistency Profiling: Robust Detection of LLM-Generated Textual Content for Multimedia Moderation |
| 作者 | Siyuan Li 等 9 位作者 |
| 机构 | 未完全披露 |
| arXiv | https://arxiv.org/abs/2605.05950 |
| 提交日期 | 2026-05-07 |
| 领域标签 | 内容审核 · AI生成内容检测 · 风格一致性 · 轻量化 · 多媒体 |
| 桶类别 | STRONG |
| 综合评分 | **63 / 100** |

---

## 方法概述 (中文)

随着 LLM 被广泛用于内容生成，平台需要可靠地区分"人类撰写"与"AI生成"内容，以维护内容生态真实性（如电商商品评论、达人内容等）。现有 AI 文本检测器依赖统计特征或模型特定启发规则，在改写（Paraphrasing）和对抗扰动下鲁棒性差。

**LiSCP（Lightweight Stylistic Consistency Profiling）**提出风格一致性剖析方法：
1. **核心思路**：通过多模态引导的改写变体，构建文本的"风格一致性画像"——AI 生成文本在风格特征上具有更强的一致性（跨改写保持不变），而人类文本风格更多变。
2. **双轨特征**：结合**离散风格特征**（词法、句法模式）与**连续语义信号**（嵌入相似度），构建鲁棒的一致性画像。
3. **对抗鲁棒性**：专门针对改写攻击设计特征稳定性验证，相比依赖单一统计特征的方法显著提升在对抗条件下的检测准确率。

---

## 故事线 (Story Arc)

> **现状不足：** 电商评论、内容平台上 AI 生成内容泛滥，现有检测器在改写/对抗扰动下准确率骤降，无法支撑大规模内容质量保障。
>
> **我们的解法：** 利用"AI文本风格一致性强于人类"这一本质特性，构建跨改写变体的风格一致性画像（LiSCP），无需大型生成模型，实现轻量鲁棒的 AI 内容检测。

---

## 创新点分析

| 维度 | 描述 |
|------|------|
| 核心创新 | 利用多模态引导改写构建风格一致性画像，从一致性角度检测 AI 内容 |
| vs. 先前工作 | 先前检测器多依赖困惑度（perplexity）等统计特征，易被改写攻击；LiSCP 利用风格稳定性，对改写具有内在鲁棒性 |
| 效率 | 轻量化设计，无需调用大型 LLM 做最终判断 |
| 局限 | 依赖多次改写生成，推理时有额外开销；对极短文本的检测效果未深入验证 |

---

## 关键指标

| 数据集 / 任务 | 指标 | LiSCP | 对比基线 |
|--------------|------|-------|---------|
| AI文本检测（无攻击） | AUROC | 高于统计基线 | GPTZero, DetectGPT |
| AI文本检测（改写攻击） | AUROC | **显著更鲁棒** | 传统检测器大幅下降 |
| 多种 LLM 生成源 | F1 | 跨 LLM 泛化 | 单模型特化检测器 |

---

## 评分分解

| 维度 | 分数 | 满分 | 说明 |
|------|------|------|------|
| Innovation | 20 | 30 | 风格一致性画像思路新颖，跨改写鲁棒性设计有价值 |
| Experimental SOTA delta | 8 | 15 | 鲁棒性改善显著，但绝对值未完整披露 |
| Experimental quality | 9 | 15 | 多LLM、多攻击类型测试，但样本规模有限 |
| Efficiency | 7 | 10 | 轻量推理，但改写生成有额外计算 |
| Generalization | 3 | 5 | 跨多种LLM生成源验证 |
| Domain relevance | 16 | 25 | 适用于电商评论/达人内容真实性检测，间接相关 |
| **Total** | **63** | **100** | |
