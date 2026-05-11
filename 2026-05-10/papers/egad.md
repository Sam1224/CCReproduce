# EGAD: Entropy-Guided Adaptive Distillation for Token-Level Knowledge Transfer

## 基本信息 / Basic Info

| 字段 | 内容 |
|---|---|
| **标题** | EGAD: Entropy-Guided Adaptive Distillation for Token-Level Knowledge Transfer |
| **arXiv ID** | [2605.01732](https://arxiv.org/abs/2605.01732) |
| **提交日期** | 2026-05-02（约） |
| **作者** | Hao Zhang, Zhibin Zhang, Guangxin Wu, Wanyi Ning, Jiafeng Guo, Xueqi Cheng |
| **机构** | 中国科学院计算技术研究所（ICT, CAS） |
| **领域桶** | WEAK |
| **综合评分** | **61 / 100** |

---

## 方法概述 (Chinese)

大语言模型（LLM）强大但部署成本高昂；知识蒸馏（KD）是将大模型知识迁移至小模型的有效方案。然而现有蒸馏方法**对所有 token 一视同仁**，忽略了不同 token 对模型决策贡献度的差异，导致知识迁移效率低下。

EGAD（**E**ntropy-**G**uided **A**daptive **D**istillation）提出基于 token 级熵的自适应蒸馏策略，从三个维度动态调整蒸馏过程：

1. **Token 级课程学习（Token-level Curriculum）**：训练初期聚焦低熵（易）token，逐步转移至高熵（难）token，模拟人类学习的渐进过程；
2. **自适应蒸馏温度（Adaptive Temperature）**：根据 token 的教师熵动态调整蒸馏温度，更好地捕捉教师模型的置信度分布；
3. **双分支架构（Dual-Branch Architecture）**：对低熵 token 采用高效的 logits-only 蒸馏，对高熵 token 进行更深层的特征级蒸馏，兼顾效率与效果。

## Method Overview (English)

EGAD proposes entropy-based adaptive LLM distillation. A token-level curriculum shifts training focus from low-entropy (easy) to high-entropy (hard) tokens. Distillation temperature adapts per token based on teacher entropy. A dual-branch architecture applies logits-only distillation to easy tokens and deeper feature-based distillation to hard tokens, balancing efficiency and knowledge transfer depth.

---

## Story Arc

**现有 LLM 蒸馏方法对所有 token 平等对待，忽视不同 token 对模型决策的贡献差异，导致知识迁移低效 → EGAD 通过熵引导的 token 级课程、自适应温度和双分支架构，精准地对高价值 token 进行深度蒸馏，提升迁移效率和学生模型性能。**

> Distilling a model is like tutoring a student — you shouldn't spend equal time on trivial and hard questions. EGAD focuses deep teaching on hard tokens (high teacher uncertainty) and breezes through easy ones.

---

## 创新性分析

1. **Token 级熵引导**：将教师的输出不确定性转化为蒸馏强度信号，是直觉合理且实现清晰的创新；
2. **课程学习+自适应温度+双分支**：三个机制协同设计，各有独立动机；
3. **计算效率**：低熵 token 的 logits-only 蒸馏降低计算开销。

**局限性**：针对 LLM 蒸馏的通用改进，与电商/内容治理领域关联需通过下游任务体现。

---

## 关键指标 / Key Metrics

| 实验设置 | 学生模型指标 | EGAD vs. 基线蒸馏 |
|---|---|---|
| 标准LLM蒸馏基准 | 多任务性能 | 优于传统蒸馏方法 |
| 效率 | 计算开销 | 低熵token节省计算 |

---

## 评分详情 / Scoring Breakdown

| 维度 | 满分 | 得分 | 说明 |
|---|---|---|---|
| 创新性 (Innovation) | 30 | 19 | Token级熵引导+课程+双分支，设计扎实但非颠覆性 |
| 实验SOTA增益 (SOTA delta) | 15 | 10 | 超越传统蒸馏基线 |
| 实验质量与消融 (Quality) | 15 | 9 | 标准蒸馏基准，消融三个机制 |
| 效率 (Efficiency) | 10 | 8 | 双分支设计提升效率 |
| 泛化性 (Generalization) | 5 | 3 | LLM通用蒸馏，跨任务 |
| 领域相关性 (Domain) | 25 | 12 | 蒸馏技术可用于电商模型压缩，但非直接贡献 |
| **总分** | **100** | **61** | — |

---

## 链接 / Links

- 论文: https://arxiv.org/abs/2605.01732
- HTML版: https://arxiv.org/html/2605.01732
