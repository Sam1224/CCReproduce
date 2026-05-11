## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **标题** | MultiSoc-4D: A Benchmark for Diagnosing Instruction-Induced Label Collapse in Closed-Set LLM Annotation of Bengali Social Media |
| **arXiv ID** | [2605.06940](https://arxiv.org/abs/2605.06940) |
| **提交日期** | 2026-05-07 |
| **作者** | (Team from NLP/AI lab, specific names not retrieved) |
| **机构** | 待补充 |
| **领域** | Data Quality · LLM Annotation · Content Moderation · Social Media NLP |
| **Bucket** | STRONG |

---

## 方法概述 / Method Summary

MultiSoc-4D 构建了一个包含 58,000+ 孟加拉语社交媒体评论的多维度标注基准，用于诊断"指令诱导标签崩溃（Instruction-Induced Label Collapse）"现象：

1. **数据集构建**：58K+ 条评论来自 6 个来源，沿四个维度标注：
   - 类别（Category）
   - 情感（Sentiment）
   - 仇恨言论（Hate Speech）
   - 讽刺（Sarcasm）
2. **标注流程**：ChatGPT、Gemini、Claude、Grok 各自独立标注不同分区，共享 20% 验证集。
3. **关键发现——"标签崩溃"**：LLM 系统性地偏向"回退标签"（Other、Neutral、No），导致高一致率但严重漏检少数类：
   - 79% 仇恨内容被误判为 No
   - 75% 讽刺内容被误判为 No
   - Fleiss' Kappa 在讽刺维度 ≈ -0.001（近乎随机一致）
4. **大规模评测**：40+ 个主流 LLM 均被测试，结果显示该偏差与架构差异无关，是 LLM 标注的系统性缺陷。

### Story Arc
> **LLM 标注被广泛用于扩展 NLP 数据集** → MultiSoc-4D 发现"指令诱导标签崩溃"：无论模型大小/架构，LLM 均系统性地漏标少数类（仇恨/讽刺），导致"标签一致性幻觉"，严重威胁基于 LLM 标注数据训练的内容审核系统可靠性。

---

## 创新性分析 / Innovation Analysis

**与先前工作对比：**
- 已有研究：发现 LLM 标注存在误差，但多属于个案研究
- 本文：首次系统性识别"指令诱导标签崩溃"作为 LLM 标注的普遍规律，并在 40+ 模型上验证

**关键创新：**
1. "Instruction-Induced Label Collapse"概念提出——LLM 的指令遵循机制使其系统性回退至安全/默认标签
2. 跨四维度标注揭示不同任务的崩溃严重程度差异（讽刺最严重）
3. 对电商违规内容自动标注（如违规商品、不实宣传检测）有直接警示意义

**可行性：** 高。数据集开放，方法论可复现。

---

## 关键指标 / Key Metrics

| LLM | 仇恨言论漏检率 | 讽刺漏检率 | Kappa (Sarcasm) |
|-----|-------------|---------|----------------|
| All tested LLMs (avg) | 79% | 75% | ≈ -0.001 |
| Human-calibrated | — | — | 显著 > 0 |

---

## 评分 / Scoring

| 维度 | 得分 | 说明 |
|------|------|------|
| Innovation | 18/30 | 新概念"标签崩溃"的提出，40+ 模型验证 |
| SOTA Delta | 8/15 | 描述性研究，无改进方法 |
| Exp Quality / Ablations | 11/15 | 4 维度×多模型全面评估 |
| Efficiency | 7/10 | 标注流程分析 |
| Generalization | 3/5 | 针对孟加拉语，跨语言泛化待验证 |
| Domain Relevance | 18/25 | 数据标注质量对内容审核系统直接影响 |
| **总分** | **65/100** | Feishu 推送 |
