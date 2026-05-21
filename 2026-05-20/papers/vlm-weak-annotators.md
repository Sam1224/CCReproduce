# Leveraging Vision-Language Models as Weak Annotators in Active Learning

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **标题** | Leveraging Vision-Language Models as Weak Annotators in Active Learning |
| **作者** | Phuong Ngoc Nguyen, Kaito Shiku, Ryoma Bise, Seiichi Uchida, Shinnosuke Matsuo |
| **机构** | Kyushu University (Japan) |
| **arXiv ID** | [2605.00480](https://arxiv.org/abs/2605.00480) |
| **提交日期** | May 1, 2026 |
| **代码** | Not yet public |
| **Bucket** | WEAK |

---

## 方法概述 / Method Overview

**EN:** Active learning reduces annotation cost by querying the most informative samples. However, even with selective querying, human annotation remains expensive. VLMs (CLIP, LLaVA, etc.) can provide free, scalable annotations — but their reliability varies dramatically with label granularity: they excel at coarse-grained labels but fail at fine-grained ones.

The proposed framework **stratifies the annotation pipeline**: human annotators provide fine-grained labels for selected samples, while VLMs supply coarse-grained weak labels for the rest. An **instance-wise label assignment** mechanism determines which modality of label (human/VLM) to use per instance. A small set of trusted fully-labeled samples is used to **model the systematic noise** in VLM-generated labels (i.e., a noise transition matrix), correcting VLM errors before training.

**ZH:** 框架将人工标注（细粒度，昂贵）与 VLM 标注（粗粒度，免费）分层结合：主动学习选择最难样本交由人工，VLM 为剩余样本提供粗粒度弱标签；实例级标签分配机制决定每个样本使用哪类标签，并用少量可信样本建模 VLM 的系统性噪声并校正。

---

## 故事主线 / Story Arc

> **现有方法的不足:** 主动学习虽减少了标注样本数，但被选样本仍需昂贵的人工细粒度标注；VLM 的免费标注未被有效利用。
>
> **我们的解决方案:** 将 VLM 定位为"弱标注者"，在主动学习框架内系统性地利用 VLM 粗粒度标注，通过噪声建模校正其误差，在相同标注预算下大幅提升模型性能。

---

## 关键指标 / Key Metrics

| Dataset | Metric | Proposed | AL Baseline |
|---------|--------|----------|-------------|
| CUB-200 (100 human labels) | Acc | ~71.4% | ~65.8% |
| FGVC-Aircraft | Acc | ~68.2% | ~62.1% |

---

## 评分 / Scoring

| 维度 | 分数 | 理由 |
|------|------|------|
| Innovation | 16/30 | 思路合理但渐进式，VLM 弱监督+主动学习组合已有先例 |
| Experimental SOTA delta | 10/15 | +5~6pp 准确率，中等提升 |
| Experimental quality | 10/15 | 两个细粒度数据集，消融完整 |
| Efficiency | 8/10 | 减少人工标注量显著 |
| Generalization | 3/5 | 细粒度视觉识别场景，对文本/多模态未验证 |
| Domain relevance | 15/25 | 适用于电商商品图像的大规模低成本标注 |
| **Total** | **62/100** | |
