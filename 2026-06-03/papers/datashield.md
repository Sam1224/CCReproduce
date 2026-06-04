# DataShield: Safety-degrading Data Filtering for LLM Benign Instruction Fine-Tuning

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | DataShield: Safety-degrading Data Filtering for LLM Benign Instruction Fine-Tuning |
| **Authors** | ZJunBo et al. |
| **Affiliation** | Not fully retrieved |
| **arXiv** | https://arxiv.org/abs/2606.00160 |
| **Submitted** | 2026-05-29 (appears in June 3, 2026 GMT+8 listing) |
| **Domain** | Data Quality · LLM Safety · Fine-tuning · Data Filtering |
| **Code** | Official: https://github.com/ZJunBo/DataShield (verified: real implementation with training scripts) |

---

## 方法概述 / Method Overview

### 问题 / Problem
即使在无害的指令微调数据集上进行 fine-tuning，LLM 的**安全对齐能力也会降低**。现有方法识别"安全降级样本"（safety-degrading samples）面临两大困境：**高计算成本**（需要大量前向传播）和**噪声过多**（误判比例高）。

Even fine-tuning on benign datasets causes LLMs to suffer degraded safety alignment. Existing methods for identifying safety-degrading samples face two issues: **high computational cost** and **excessive noise** (high false positive rate).

### 方法 / Method

**DataShield** 通过量化每个样本对模型服从行为（compliance behavior）的贡献来识别危险样本，核心三步骤：

**Step 1 — 合规向量提取（Compliance Vector Extraction）：**
从 LLM 的特定层提取**合规向量（Compliance Vector）**，捕捉模型的整体服从倾向（compliance tendency）。这个向量代表了模型在安全与顺从之间的内在权衡。

**Step 2 — 合规感知评分（Compliance-Aware Score, CAS）：**
计算每个训练样本对合规向量方向的贡献，并通过 CAS 自动确定最优安全关键层（safety-critical layer）。CAS 反映了该样本"推动模型变得更顺从"的程度。

**Step 3 — 安全降级样本过滤（Safety-degrading Sample Filtering）：**
根据 CAS 分数过滤高风险样本，保留对安全对齐无负面影响的数据子集。

**关键发现：** 开放式问答（open-ended QA）类样本更可能触发安全降级，对应的响应往往更长。

**Story Arc:** "良性数据也会损害 LLM 安全 → DataShield 用合规向量高效识别并过滤危险样本"

*Benign data still degrades LLM safety → DataShield uses compliance vectors to efficiently identify and filter safety-degrading samples.*

---

## 创新性分析 / Innovation

1. **合规行为作为代理信号**：利用 LLM 内部的合规倾向向量作为数据风险评分的代理，无需直接进行安全测试，计算高效。
2. **自动层选择**：CAS 自动确定最优安全关键层，无需人工调参。
3. **无需安全标注**：纯粹从良性微调数据中识别有害样本，不依赖任何安全/非安全标签。
4. **实证发现**：揭示了"开放式 QA + 长响应"与安全降级的相关性，为数据工程实践提供了可操作指导。

---

## 关键指标 / Key Metrics

| Model | Dataset | Safety Before | Safety After DataShield |
|-------|---------|---------------|------------------------|
| Llama3-8B | Alpaca | Degraded | Preserved |
| Llama3.1-8B | Dolly | Degraded | Preserved |
| Qwen2.5-7B | Alpaca | Degraded | Preserved |

*Validated across three model families and two benign fine-tuning datasets.*

---

## 评分 / Scoring

| Dimension | Max | Score | Justification |
|-----------|-----|-------|---------------|
| Innovation | 30 | 22 | Novel compliance vector approach; automatic layer selection; zero safety annotation requirement |
| SOTA Delta | 15 | 11 | Validated across three model families (Llama3, Llama3.1, Qwen2.5); two datasets |
| Exp. Quality | 15 | 11 | Multi-model, multi-dataset evaluation; open-ended QA finding |
| Efficiency | 10 | 8 | Low computational cost vs. existing methods; official code available |
| Generalization | 5 | 4 | Tested on multiple model families |
| Domain Relevance | 25 | 18 | Data quality/filtering for LLM training directly relevant to large-scale data labeling and content governance systems |
| **Total** | **100** | **74** | |

---

## 与先前工作的对比 / Comparison with Prior Work

| Method | Annotation Needed | Cost | Noise Level |
|--------|------------------|------|-------------|
| Safety-aware fine-tuning | Yes (safety labels) | High | Medium |
| Gradient-based filtering | No | Very High | High |
| **DataShield** | **No** | **Low** | **Low** |

---

## 电商/内容治理相关性 / E-commerce & Governance Relevance

DataShield 直接适用于：
- **大规模数据标注质量控制**：在电商场景中，从海量用户生成内容（UGC）训练 LLM 时，过滤可能损害安全对齐的样本
- **内容治理 LLM 的持续微调**：违规检测模型在持续学习新数据时，防止安全对齐退化
- **达人内容审核模型维护**：在用新内容微调审核模型时保持安全性

DataShield applies to: quality control in large-scale UGC-based LLM training, safety maintenance during continuous fine-tuning of content moderation models, and influencer content review model maintenance.
