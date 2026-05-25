# DetectRL-X: Towards Reliable Multilingual and Real-World LLM-Generated Text Detection

## 基本信息 / Basic Info

| 字段 | 内容 |
|---|---|
| **Title** | DetectRL-X: Towards Reliable Multilingual and Real-World LLM-Generated Text Detection |
| **Authors** | Junchao Wu, Yefeng Liu, Chenyu Zhu, Hao Zhang, Zeyu Wu, Tianqi Shi, Yichao Du, Longyue Wang, Weihua Luo, Jinsong Su, Derek F. Wong |
| **Affiliation** | Multiple institutions (Xiamen University, Tencent, etc. inferred) |
| **arXiv** | [2605.15518](https://arxiv.org/abs/2605.15518) |
| **Submitted** | 15 May 2026 (revised 19 May 2026) |
| **Domain** | AIGC Detection · Content Governance · Multilingual NLP |
| **Bucket** | WEAK |
| **Code** | No official code mentioned |

---

## 分数 / Score

| 维度 | 分数 | 满分 |
|---|---|---|
| Innovation | 18 | 30 |
| Experimental SOTA delta | 10 | 15 |
| Experimental quality / ablations | 11 | 15 |
| Efficiency | 4 | 10 |
| Generalization | 4 | 5 |
| Domain relevance (ecom + governance) | 14 | 25 |
| **Total** | **61** | **100** |

**Justification**: Comprehensive multilingual AIGC detection benchmark covering 8 languages and realistic AI-assisted writing scenarios. Important for content governance but lacks direct e-commerce focus. The inclusion of AI-assisted writing (polishing, expanding, condensing) makes it highly relevant to detecting LLM-generated product descriptions and influencer marketing copy. Domain relevance limited to the text modality.

---

## 方法概述 / Method Summary

随着 LLM 在商业场景中被大量用于生成内容（包括商品描述、评论、营销文案），如何可靠地检测 LLM 生成文本变得日益紧迫。现有 LLM 文本检测基准存在三大缺陷：① 仅覆盖英语，忽视多语言商业场景；② 主要测试完全机器生成的文本，忽视人机协作写作（AI 辅助润色、扩写、压缩）；③ 缺乏商业场景高风险领域的覆盖。

**DetectRL-X** 构建了迄今最全面的多语言 LLM 文本检测基准：

1. **语言覆盖**: 8 种语言（中、英、西、法、德、日、韩、俄），覆盖商业场景主流语种。
2. **领域覆盖**: 6 个易被 LLM 滥用的高风险领域（学术写作、新闻、社交媒体、电商评论、医疗、法律）。
3. **文本生成方式**: 4 种主流商业 LLM 生成文本，外加 AI 辅助写作操作（润色 Polishing / 扩写 Expanding / 压缩 Condensing）。
4. **评估维度**: 8 个检测维度，包括跨域泛化、跨语言泛化、攻击鲁棒性、低资源场景等。

**Story Arc**: LLM 生成文本检测在多语言、真实写作场景下的可靠性严重不足 → DetectRL-X 通过构建 8 维、8 语言、6 领域的全面基准，系统揭示当前检测器的瓶颈。

---

## 创新性分析 / Innovation Analysis

**与 prior work 的对比**:
- DetectRL（前作）：英语为主，场景有限。
- MGTBench、RAID：缺乏 AI 辅助写作场景。
- **DetectRL-X 独特性**:  
  (a) 首次全面覆盖 AI 辅助写作（非完全机器生成）——最接近真实商业使用场景；  
  (b) 多语言覆盖远超现有基准；  
  (c) 商业高风险领域（含电商评论）的系统性评估。

---

## 关键指标 / Key Metrics

| 检测器类型 | 跨语言泛化 | AI辅助写作检测 | 整体AUC |
|---|---|---|---|
| 零样本 LLM 检测 | 较弱 | 低 | <0.70 |
| 微调分类器 | 单语言强 | 中等 | ~0.80 |
| 跨语言预训练模型 | 最强 | 较强 | 最高 |

---

## 与电商内容生态的关联

- **虚假商品评论检测**: LLM 生成的虚假好评/差评是电商平台的重大治理挑战，DetectRL-X 的电商评论领域测试直接相关。
- **AI 生成商品描述识别**: 卖家使用 LLM 批量生成商品标题/详情页，可能包含夸大宣传，检测工具有助于内容质量审核。
- **达人营销文案合规**: AI 辅助润色的营销文案可能规避人工审核，本基准的 AI 辅助写作场景覆盖正是此类风险。

---

## 论文链接

- arXiv: https://arxiv.org/abs/2605.15518
