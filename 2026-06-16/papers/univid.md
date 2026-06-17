# UNIVID: Unified Vision-Language Model for Video Moderation

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **标题** | UNIVID: Unified Vision-Language Model for Video Moderation |
| **作者** | Kejuan Yang*, Yizhuo Zhang*, Mingyuan Du, Yue Zhang, Dixin Zheng, Kaili Zhao, Yang Xiao, Hanzhong Liang, Kenan Xiao |
| **机构** | ByteDance |
| **链接** | https://arxiv.org/abs/2606.05748 |
| **arXiv ID** | 2606.05748 |
| **提交日期** | June 4, 2026 (appeared in June 5–6 arXiv listing) |
| **Bucket** | STRONG |
| **Code** | `code/UNIVID/` |

---

## 方法概述 / Method Overview

**中文：**  
UNIVID是字节跳动提出的首个专为全球规模视频内容审核设计的统一视觉-语言模型。其核心创新在于**策略感知字幕生成（Policy-Aware Captioning）**：不同于传统黑盒分类器直接输出标签，UNIVID生成符合平台安全策略的结构化文本描述作为中间表示，该描述既可被人工审核者理解验证，又可作为下游多任务决策的统一输入。

系统采用三阶段级联架构：
1. **风险过滤器（Risk Filter）**：多模态风险漏斗，融合UNIVID字幕+视觉信号，过滤潜在高风险视频
2. **审核执行器（Moderation Actor）**：包含UNIVID-Lite（轻量快速决策）和UNIVID-RAG（基于历史违规案例的检索增强召回）
3. **趋势治理（Trend Governance）**：缓存UNIVID嵌入向量，通过自适应聚类检测新兴风险趋势

**English:**  
UNIVID (ByteDance) is the first unified VLM for video content moderation at global scale. Instead of black-box classifiers, it generates **policy-aware captions** as interpretable intermediate representations. These captions enable human-verifiable decisions and multi-task reusability. The moderation pipeline has three cascaded stages: (A) Risk Filter (multimodal funnel), (B) Moderation Actor (UNIVID-Lite + UNIVID-RAG), (C) Trend Governance (embedding-based emerging risk detection).

---

## 故事弧线 / Story Arc

**现有方法不足 →** 传统视频审核依赖黑盒分类器，无法解释决策、难以适应策略更新，且商业VLM存在安全护栏拒绝（guardrail refusal）和策略对齐不足的问题。  
**本文设计 →** 专门训练的策略感知UNIVID模型，结合专家标注+合成数据训练方案，生成可解释策略字幕，支持多任务复用、人工审核和趋势追踪。

---

## 创新性 / Innovation

1. **策略感知字幕作为通用中间表示**：将传统多个分类器统一为单一文本生成任务，字幕同时服务于决策、解释、召回和趋势检测。
2. **UNIVID-RAG**：利用历史违规案例库，通过检索增强补充规则盲区，解决新型违规召回问题。
3. **Trend Governance**：嵌入缓存+聚类自适应检测新兴违规趋势，无需人工标注新类别即可自动发现。
4. **训练数据方案**：专家精炼标注+合成数据结合，克服商业VLM因安全对齐而拒绝生成违规描述的问题。

**与前工作的差异：** 此前视频审核模型均为任务特定分类器，无法迁移；UNIVID首次实现跨任务统一，并利用语言空间做策略对齐。

---

## 关键指标 / Key Metrics

*具体指标待论文公开详细数值；已知信息如下：*

| 数据集/场景 | 指标 | 说明 |
|-------------|------|------|
| ByteDance内部视频审核数据 | 多任务统一审核指标 | 优于单任务专用模型 |
| 字幕质量 | 可解释性/策略一致性 | 专家评估通过率提升 |
| 趋势检测 | 新兴违规识别 | 自动发现无需手动标注 |

---

## 评分 / Scoring

| 维度 | 分数 | 满分 | 说明 |
|------|------|------|------|
| 创新性 (Innovation) | 25 | 30 | 字幕中间表示+三阶段架构，系统性创新 |
| 实验SOTA增量 (SOTA Delta) | 12 | 15 | 缺乏公开对比数值 |
| 实验质量/消融 (Exp Quality) | 12 | 15 | ByteDance内部验证，消融完整 |
| 效率 (Efficiency) | 8 | 10 | UNIVID-Lite保证推理效率 |
| 泛化性 (Generalization) | 4 | 5 | 多任务统一，迁移性强 |
| 领域相关性 (Domain Relevance) | 22 | 25 | 视频内容审核+趋势治理，高度相关 |
| **Total** | **83** | **100** | |

---

## 复现代码位置

`code/UNIVID/` — 完整复现代码（PyTorch实现）

---

## 参考链接

- arXiv: https://arxiv.org/abs/2606.05748  
- ByteDance GitHub: https://github.com/bytedance
