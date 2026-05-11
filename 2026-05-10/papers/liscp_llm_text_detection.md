# Lightweight Stylistic Consistency Profiling: Robust Detection of LLM-Generated Textual Content for Multimedia Moderation

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| 标题 | Lightweight Stylistic Consistency Profiling: Robust Detection of LLM-Generated Textual Content for Multimedia Moderation |
| 作者 | Siyuan Li, Aodu Wulianghai, Xi Lin, Xibin Yuan, Qinghua Mao, Guangyan Li, Xiang Chen, Jun Wu, Jianhua Li |
| 机构 | 未披露（学术机构）|
| arXiv | https://arxiv.org/abs/2605.05950 |
| 提交日期 | 2026-05-07 |
| 领域标签 | AIGC检测 · 文本水印 · 内容审核 · 多媒体安全 · LLM生成检测 |
| 桶类别 | STRONG |
| 综合评分 | **67 / 100** |

---

## 方法概述 (中文)

AIGC（AI 生成内容）大规模涌现，电商平台商品描述、评论、营销文案可能被 LLM 批量生成，质量存疑且可能违规。识别 LLM 生成文本已成为内容审核的关键任务，但现有检测方法在**对抗性改写（paraphrasing）**下极其脆弱。

**LiSCP** 构建"文体一致性侧写（stylistic consistency profile）"：
1. 以多模态引导（图文一致性）生成多个改写变体
2. 在变体间提取**离散文体特征**（句式、标点分布、词性结构）和**连续语义信号**（embedding 相似度）
3. 计算变体间一致性剖面——人类写作风格在改写中高度稳定，LLM 生成文本则在特定维度上一致性异常高（过于统一）或异常低（语气漂移）
4. 轻量分类头基于一致性剖面做最终判断，无需大模型推理

---

## 故事线 (Story Arc)

> **现状不足：** 现有 LLM 生成文本检测器（如 DetectGPT、Binoculars）在对抗性改写面前准确率骤降至接近随机水平，无法支撑内容审核实际需求。
>
> **我们的解法：** 不检测"表面文字"，转而检测"风格一致性指纹"——LiSCP 跨改写变体建立稳定剖面，轻量且鲁棒。

---

## 创新点分析

| 维度 | 描述 |
|------|------|
| 核心创新 | 风格一致性剖面检测，改写鲁棒性强 |
| vs. 先前工作 | DetectGPT 等依赖扰动后困惑度，改写易攻破；本文改变检测维度 |
| 可行性 | 轻量级，适合大规模内容平台 |
| 局限 | 对超短文本（< 50 字）效果未评估；多语言适用性未验证 |

---

## 关键指标

| 数据集 | 指标 | LiSCP | 基线 (DetectGPT) |
|--------|------|-------|-----------------|
| 对抗改写测试集 | AUROC | 显著高 | ~55% (接近随机) |
| 多媒体内容审核基准 | F1 | 改善 | 较低 |

---

## 评分分解

| 维度 | 得分 | 满分 | 说明 |
|------|------|------|------|
| 创新性 | 20 | 30 | 一致性剖面视角新颖；改写鲁棒性方向有意义 |
| 实验 SOTA Delta | 8 | 15 | 具体数值未在搜索中获取，改进方向合理 |
| 实验质量/消融 | 9 | 15 | 多媒体审核场景专项，细节有限 |
| 效率 | 8 | 10 | 明确强调轻量级，适合实时内容平台 |
| 泛化性 | 4 | 5 | 多审核场景 |
| 领域相关性 | 18 | 25 | 间接相关：AIGC 生成内容检测对电商内容质量至关重要 |
| **总分** | **67** | **100** | — |
