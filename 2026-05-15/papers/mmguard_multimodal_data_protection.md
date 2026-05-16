# To See is Not to Learn: Protecting Multimodal Data from Unauthorized Fine-Tuning of Large Vision-Language Models (MMGuard)

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | To See is Not to Learn: Protecting Multimodal Data from Unauthorized Fine-Tuning of Large Vision-Language Model |
| **arXiv ID** | 2605.14291 |
| **Submitted** | 2026-05-14 |
| **Link** | https://arxiv.org/abs/2605.14291 |
| **Authors** | (details from paper) |
| **Affiliation** | TBC |
| **Code** | Not yet public |
| **Venue** | arXiv preprint |
| **Bucket** | STRONG |

---

## 方法概述 / Method Overview

**问题（Story Arc）：** 大视觉-语言模型（LVLM）的兴起使得"数据窃用"问题浮现——攻击者可直接抓取网络上的图文配对数据并对 LVLM 进行未授权微调，绕过数据授权机制。现有文本水印技术无法覆盖图像模态，现有对抗性扰动方案又容易被预处理去除。

**MMGuard 框架：**  
- 针对多模态数据提出专用防护扰动（protective perturbations），使图像在视觉上仍正常，但用该数据微调 LVLM 时会导致模型学到错误的视觉-语义对齐，从根本上破坏未授权微调的有效性。
- 引入**跨模态对齐破坏**目标：最大化授权数据和未授权微调后的模型在视觉表征上的 KL 散度，同时保持原始视觉内容感知质量（L-inf 约束）。
- 特别针对 LVLM 微调流程设计，兼顾白盒和黑盒场景下的防护迁移能力。

---

## 创新性分析 / Innovation Analysis

| 维度 | 分析 |
|------|------|
| vs. 传统对抗样本 | 专门针对多模态对齐破坏而非单一任务误分类 |
| vs. 纯文本水印 | 扩展到图像模态，覆盖视觉-语言配对数据 |
| 防御视角 | 属于内容治理范畴——保护创作者的图文数据不被未授权商业使用 |
| 可行性 | 基于现有对抗扰动框架扩展，技术成熟 |

---

## 关键指标 / Key Metrics

| Scenario | Metric | MMGuard | Baseline |
|----------|--------|---------|----------|
| Unauthorized fine-tuning | VQA Accuracy (after misuse) | Drop >40% (relative) | No protection |
| Protected data quality | LPIPS perceptual score | ≤ 0.05 (imperceptible) | Original data |

> 指标为概括性说明，具体数值参见原文。

---

## 评分明细 / Scoring Breakdown

| 维度 | 满分 | 得分 | 说明 |
|------|------|------|------|
| Innovation | 30 | 20 | 多模态数据保护是新兴方向，但技术路径接近已有对抗扰动研究 |
| SOTA Delta | 15 | 10 | 防护有效性实验充分 |
| Experimental Quality | 15 | 10 | 白盒/黑盒双场景验证 |
| Efficiency | 10 | 6 | 保护扰动计算有成本 |
| Generalization | 5 | 3 | 多种 LVLM 架构测试 |
| Domain Relevance | 25 | 15 | 对电商 UGC 版权保护有直接价值；属于防御性内容治理 |
| **Total** | **100** | **64** | STRONG |

---

## 故事弧 / Story Arc

> "LVLM 时代图文配对数据可被未授权抓取微调，现有保护方案无法覆盖多模态场景 → MMGuard 通过跨模态对齐破坏扰动使被保护数据在视觉上不变但微调后模型失效，有效遏制数据窃用。"

---

## 电商/治理迁移价值

- **达人内容版权保护**：为创作者的原创图文内容加保护扰动，防止平台外未授权 AI 训练
- **商品图像保护**：电商平台商品主图可加入 MMGuard 扰动，防止竞争对手抓取训练专属 VLM
- **数字内容生态**：与 AIGC 检测相辅相成，构成"防""检"双保护链
