# VANGUARD: Reasoning-Guided Grounding for Video Anomaly Detection via MLLMs

## 基本信息 / Basic Info

| 字段 | 内容 |
|---|---|
| **标题** | Reasoning-Guided Grounding: Elevating Video Anomaly Detection through Multimodal Large Language Models |
| **arXiv ID** | [2605.02912](https://arxiv.org/abs/2605.02912) |
| **提交日期** | 2026-04-07（更新版出现于2026-05月） |
| **作者** | Sakshi Agarwal, Aishik Konwer, Ankit Parag Shah |
| **机构** | 未完全披露 |
| **领域桶** | STRONG |
| **综合评分** | **71 / 100** |

---

## 方法概述 (Chinese)

视频异常检测（VAD）传统上被建模为二分类或离群点检测，缺乏可解释推理和精确的空间定位能力。现有视觉-语言模型（VLM）在场景理解方面能力强大，但空间定位（bounding box 生成）常产生幻觉或几何不合理的结果。

VANGUARD（Video Anomaly iNtelliGence via reAsoning and groUnding with RewarDeD training）将异常分类、空间定位与链式推理（Chain-of-Thought）统一在单一 VLM 框架内，通过三阶段课程式训练（curriculum training）逐步叠加学习目标：

1. **分类热身（Classifier Warmup）**：在冻结 backbone 特征上训练异常分类器；
2. **LoRA 空间定位适配（LoRA-adapted Spatial Grounding）**：通过 LoRA 微调学习精确的异常区域定位；
3. **链式思维生成（Chain-of-Thought Generation）**：学习生成可解释的推理过程说明。

在 UCF-Crime 基准上，VANGUARD 以 94% ROC-AUC、84% F1 达到 SOTA，同时提供可解释的 CoT 说明和空间定位能力。

## Method Overview (English)

VANGUARD unifies anomaly classification, spatial grounding, and chain-of-thought reasoning within a single VLM through a three-stage curriculum: (1) classifier warmup on frozen backbone features, (2) LoRA-adapted spatial grounding to learn precise anomaly localization, (3) chain-of-thought generation for interpretable reasoning. Achieves 94% ROC-AUC and 84% F1 on UCF-Crime while producing both CoT explanations and spatial bounding boxes — capabilities absent from prior VAD methods.

---

## Story Arc

**视频异常检测缺乏可解释推理和精确空间定位，VLM 的定位能力又常出现幻觉 → VANGUARD 通过三阶段课程训练将分类、定位、推理统一在单一 VLM 中，在 UCF-Crime 上以 94% ROC-AUC 达到 SOTA 并提供可解释输出。**

> Traditional VAD gives a binary verdict with no explanation. VANGUARD teaches a VLM to say "this person is running suspiciously toward the exit (highlighted bounding box) because..." — combining classification, grounding, and reasoning.

---

## 创新性分析

1. **三合一统一框架**：分类+定位+推理三能力在单模型中统一，而非三个独立系统；
2. **课程式训练策略**：分阶段解锁学习目标避免任务干扰，是稳健的工程设计；
3. **VAD+CoT 的可解释性**：视频异常检测加 CoT 推理在本领域是新颖结合；
4. **VLM 幻觉问题的针对性处理**：LoRA 空间定位适配专门解决 VLM 定位幻觉问题。

**与先前工作的差异**：传统 VAD 方法（如 RTFM、CLIP-TSA）仅做分类，无定位无推理；VANGUARD 是首个在 VAD 中结合三者的工作（据其声明）。

**与电商内容治理的关联**：可迁移至直播内容异常检测（违禁商品展示、违规行为识别），为内容审核提供可解释的推理链。

---

## 关键指标 / Key Metrics

| 数据集 | 指标 | VANGUARD | 最佳基线 |
|---|---|---|---|
| UCF-Crime | ROC-AUC | **94%** | ~91% |
| UCF-Crime | F1 | **84%** | ~78% |
| — | 空间定位 | 支持 | 不支持 |
| — | CoT推理 | 支持 | 不支持 |

---

## 评分详情 / Scoring Breakdown

| 维度 | 满分 | 得分 | 说明 |
|---|---|---|---|
| 创新性 (Innovation) | 30 | 21 | 三合一统一+课程训练，对VAD领域是重要创新 |
| 实验SOTA增益 (SOTA delta) | 15 | 12 | UCF-Crime 94% ROC-AUC，显著提升 |
| 实验质量与消融 (Quality) | 15 | 11 | 单基准，消融三阶段贡献 |
| 效率 (Efficiency) | 10 | 6 | LoRA参数高效但VLM推理仍较重 |
| 泛化性 (Generalization) | 5 | 3 | 单数据集验证 |
| 领域相关性 (Domain) | 25 | 18 | 视频异常检测可迁移至直播内容审核 |
| **总分** | **100** | **71** | — |

---

## 链接 / Links

- 论文: https://arxiv.org/abs/2605.02912
- HTML版: https://arxiv.org/html/2605.02912
