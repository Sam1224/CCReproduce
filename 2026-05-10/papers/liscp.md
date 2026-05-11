# LiSCP: Lightweight Stylistic Consistency Profiling for Multimedia Moderation

## 基本信息 / Basic Info

| 字段 | 内容 |
|---|---|
| **标题** | Lightweight Stylistic Consistency Profiling: Robust Detection of LLM-Generated Textual Content for Multimedia Moderation |
| **arXiv ID** | [2605.05950](https://arxiv.org/abs/2605.05950) |
| **提交日期** | 2026-05-07 |
| **作者** | Siyuan Li et al. (9位作者) |
| **机构** | 未在摘要中公开披露 |
| **领域桶** | STRONG |
| **综合评分** | **77 / 100** |

---

## 方法概述 (Chinese)

随着大语言模型（LLM）在内容创作中的广泛应用，区分人类撰写与 LLM 生成内容已成为多媒体内容审核的关键任务。现有检测器依赖统计特征或模型特定启发式方法，在对抗性改写（paraphrasing）和对抗样本攻击下鲁棒性不足，解释性也较差。

LiSCP（**Li**ghtweight **S**tylistic **C**onsistency **P**rofiling）构建一致性档案（consistency profile），融合：
- **离散文体特征**（句法、标点、词汇多样性等）
- **连续语义信号**（基于嵌入模型的语义相似度）

关键创新在于通过**多模态引导的改写变体（multimodal-guided paraphrased variants）**进行特征稳定性检测：LLM 生成文本在文体层面具有高度一致性，即使经过改写其文体轮廓仍保持稳定；而人类写作在改写后文体特征变化较大。LiSCP 利用这一稳定性差异作为识别信号，实现对改写攻击的鲁棒性。

## Method Overview (English)

LiSCP builds a consistency profile combining discrete stylistic features (syntax, punctuation, lexical diversity) with continuous semantic signals, validated against multimodal-guided paraphrased variants. LLM-generated text maintains stylistic consistency under paraphrasing while human text varies — LiSCP exploits this gap for robust, interpretable detection. Designed to be lightweight for deployment in multimedia moderation pipelines.

---

## Story Arc

**现有 LLM 文本检测器依赖统计或模型特定启发式，对改写和对抗攻击脆弱 → LiSCP 构建文体一致性档案，利用 LLM 生成文本在改写下文体稳定的特性，实现对攻击鲁棒且轻量可解释的多媒体内容审核检测。**

> If an LLM writes something, its stylistic fingerprint persists even after paraphrasing — humans naturally vary their style. LiSCP profiles this stability difference for robust detection.

---

## 创新性分析

1. **基于稳定性的检测范式**：不直接判断文体特征"像不像 LLM"，而是检测改写前后的**稳定性**，这是对对抗攻击的根本性防御；
2. **多模态引导改写**：利用多模态上下文指导改写生成，增加了对跨模态对抗的鲁棒性；
3. **离散+连续特征融合**：一致性档案同时捕获词法层面和语义层面的信息，比单一特征更全面；
4. **轻量设计**：面向实际多媒体审核管线，推理效率是核心约束。

**与先前工作的差异**：GPT-Sentinel、Binoculars 等方法依赖模型困惑度或统计偏差，对改写脆弱；LiSCP 的稳定性检测是方向性创新。

---

## 关键指标 / Key Metrics

| 数据集 | 攻击类型 | LiSCP | 最佳基线 |
|---|---|---|---|
| 多媒体内容审核基准 | 无攻击 | 高准确率 | 相当 |
| 多媒体内容审核基准 | 改写攻击 | **鲁棒性显著更优** | 大幅下降 |
| 多媒体内容审核基准 | 多种对抗操纵 | 稳定 | 不稳定 |

---

## 评分详情 / Scoring Breakdown

| 维度 | 满分 | 得分 | 说明 |
|---|---|---|---|
| 创新性 (Innovation) | 30 | 21 | 文体稳定性检测范式新颖，多模态引导改写独特 |
| 实验SOTA增益 (SOTA delta) | 15 | 11 | 对抗场景鲁棒性显著，无攻击场景持平 |
| 实验质量与消融 (Quality) | 15 | 11 | 多种攻击场景测试，消融较充分 |
| 效率 (Efficiency) | 10 | 8 | 轻量设计，适合部署 |
| 泛化性 (Generalization) | 5 | 4 | 多媒体多场景 |
| 领域相关性 (Domain) | 25 | 22 | LLM内容检测、多媒体审核、内容治理 |
| **总分** | **100** | **77** | — |

---

## 链接 / Links

- 论文: https://arxiv.org/abs/2605.05950
- HTML版: https://arxiv.org/html/2605.05950
