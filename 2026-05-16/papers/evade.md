# EVADE: Multimodal Benchmark for Evasive Content Detection in E-Commerce Applications

## 基本信息 / Basic Info

| Field | Value |
|-------|-------|
| **Title** | EVADE: Multimodal Benchmark for Evasive Content Detection in E-Commerce Applications |
| **Authors** | (Research team — EVADE-Bench) |
| **Affiliation** | (Chinese e-commerce research, submitted to OpenReview ICLR track) |
| **arXiv ID** | [2505.17654](https://arxiv.org/abs/2505.17654) |
| **Submission Date** | May 2025 |
| **Domain Tags** | `#e-commerce` `#content-governance` `#violation-detection` `#VLM` `#benchmark` `#evasive-content` `#Chinese` |
| **Total** | **74 / 100** |

---

## 故事弧线 / Story Arc

**现有问题:** 电商平台越来越依赖 LLM/VLM 检测违规商品内容。然而，大量商家通过"**擦边内容（Evasive Content）**"规避检测——即表面符合平台规则、实则隐含违规声明的文本或图像（如"帮助身体自然增高"代替"增高产品"）。与对抗性攻击不同，擦边内容利用**政策歧义和上下文模糊性**，更难检测。**现有模型和基准均未专门针对此类场景。**

**设计方案:** 构建 **EVADE** — 首个专家精标的中文多模态电商擦边内容检测基准，对 26 个主流 LLM/VLM 进行系统评测，揭示当前最优模型在此场景下的显著性能缺口。

---

## 方法概述 / Method Overview

### 数据集构成

```
EVADE Dataset
├── Text samples: 2,833 annotated entries
│   ├── Explicit violations (clear policy breach)
│   ├── Evasive content (implicit / borderline)
│   └── Compliant content
│
└── Images: 13,961 annotated samples
    ├── Product categories (6 high-risk domains):
    │   ├── Body shaping / weight loss
    │   ├── Height growth
    │   ├── Health supplements
    │   ├── Skin whitening
    │   ├── Sexual health products
    │   └── Medical devices (OTC)
    └── Labels: violation type, severity, evasion strategy
```

### 双任务评测设计

**Task 1 — Single-Violation:** 每条规则单独测试，细粒度推理能力评估  
**Task 2 — All-in-One:** 所有相关规则合并为统一指令，长上下文推理评估

> Key finding: All-in-One 设置显著缩小模型间 partial-match 与 full-match 精度差距，说明清晰的规则定义有助于对齐人类与模型判断。

### 评测规模
- 26 个主流 LLM/VLM 参与评测（GPT系列、Claude、Qwen、InternVL 等）
- 核心发现：即使 SOTA 模型也频繁误分类擦边内容

---

## 关键指标 / Key Metrics

| Finding | Detail |
|---------|--------|
| Models evaluated | 26 mainstream LLMs/VLMs |
| Performance gap | Significant: even SOTA models frequently misclassify evasive samples |
| All-in-One vs Single-Violation | All-in-One narrows the partial/full-match gap — clearer rules improve model-human alignment |
| Best model accuracy | Below 70% on evasive categories (extrapolated from description) |

---

## 创新性分析 / Innovation Analysis

- **全球首个**面向电商擦边内容的中文多模态基准
- 擦边内容与传统对抗样本的根本区别在于：利用政策模糊性而非模型弱点，更贴近实际商业攻防场景
- 双任务设计揭示了规则表达方式对模型性能的关键影响
- 6 个高风险类目覆盖中国电商最常见违规领域

**局限性:** 当前为静态基准，不覆盖商家持续更新擦边策略的动态场景。

---

## 评分细项 / Scoring Breakdown

| Dimension | Score | Max | Justification |
|-----------|-------|-----|---------------|
| Innovation | 21 | 30 | 首个中文电商擦边内容多模态基准；任务设计精巧 |
| Experimental SOTA Delta | 10 | 15 | 26 模型系统评测；揭示性能缺口有重要价值 |
| Experimental Quality | 11 | 15 | 专家精标；双任务设计；ICLR OpenReview |
| Efficiency | 5 | 10 | 基准论文，无模型效率贡献 |
| Generalization | 4 | 5 | 覆盖 6 个高风险类目 |
| Domain Relevance | 23 | 25 | 直接针对电商违规内容检测，极高相关性 |
| **Total** | **74** | **100** | |
