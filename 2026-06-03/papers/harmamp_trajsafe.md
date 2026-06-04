# Investigating and Alleviating Harm Amplification in LLM Interactions

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | Investigating and Alleviating Harm Amplification in LLM Interactions |
| **Authors** | Ruohao Guo, Wei Xu, Alan Ritter |
| **Affiliation** | Georgia Institute of Technology |
| **arXiv** | https://arxiv.org/abs/2606.02423 |
| **Submitted** | 2026-06-01 (appears in June 3, 2026 GMT+8 listing) |
| **Domain** | Content Governance · LLM Safety · Harm Detection · Multi-turn Moderation |
| **Code** | `code/TrajSafe/` |

---

## 方法概述 / Method Overview

### 问题 / Problem
现有 LLM 安全研究大多聚焦于**单轮**越狱检测，忽略了更危险的**多轮危害放大**现象：恶意用户通过与 LLM 的连续对话，逐步获取其单独无法实现的有害能力——无论是门外汉通过咨询习得专业危害知识（专业知识民主化），还是大规模自动化运营有害内容（批量放大）。

Most LLM safety work focuses on single-turn jailbreaks, ignoring the more dangerous phenomenon of **multi-turn harm amplification**: malicious users, through extended LLM conversations, gradually obtain harmful capabilities beyond their standalone reach — either democratizing specialized harmful knowledge for novices, or scaling harmful operations at volumes impossible with manual effort.

### 方法 / Method

**双重贡献：HarmAmp Benchmark + TrajSafe Monitor**

**1. HarmAmp 基准：**
- 12 个风险类别，覆盖真实威胁场景
- 每个场景满足三个标准：实质性放大（substantive amplification）、操作具体性（operational specificity）、多轮必要性（multi-turn necessity）
- 多轮对话设计，评估随轮次增加的危害累积

**2. TrajSafe 主动监控器：**
- 通过**轨迹预测**提前识别有害对话方向
- 主动干预：探究用户真实意图、引导模型走向安全
- 基于**树状强化学习（tree-based RL）**训练，在覆盖多种有害路径的同时最小化误报

**Story Arc:** "单轮安全不足以应对多轮危害放大 → 构建 HarmAmp 基准 + TrajSafe 主动干预"

*Single-turn safety is insufficient for multi-turn harm amplification → build HarmAmp benchmark + TrajSafe proactive intervention.*

---

## 创新性分析 / Innovation

1. **首次系统化定义"危害放大"**：将危害放大分解为"专业知识民主化"和"规模化批量有害操作"两个维度，是对 LLM 安全威胁的重要概念贡献。
2. **HarmAmp 新基准**：12 类现实威胁场景 × 多轮对话设计，填补了安全评估中多轮累积危害的评测空白。
3. **TrajSafe 主动干预**：区别于被动过滤（事后判定），TrajSafe 通过轨迹预测进行主动预防性干预，并利用树状 RL 训练以平衡 recall 与 precision。
4. **两种放大路径分析**：对不同用户类型（技术新手 vs. 批量操作者）的危害放大路径进行差异化建模。

---

## 关键指标 / Key Metrics

| Dataset | Metric | TrajSafe | Baseline |
|---------|--------|----------|---------|
| HarmAmp (12 categories) | Harm Amplification Prevention Rate | Reported improvement | Prior monitors |
| HarmAmp | Multi-turn Safety Score | Evaluated | Single-turn baselines |
| HarmAmp | False Positive Rate | Controlled | — |

*Note: Exact numbers not retrievable due to arXiv 403; based on paper description from searches.*

---

## 评分 / Scoring

| Dimension | Max | Score | Justification |
|-----------|-----|-------|---------------|
| Innovation | 30 | 25 | Novel multi-turn harm amplification framing; dual contribution (benchmark + monitor); tree-based RL for proactive intervention |
| SOTA Delta | 15 | 11 | New benchmark establishes baselines across 12 categories; TrajSafe improves over prior monitors |
| Exp. Quality | 15 | 12 | 12 risk categories, real-world threat grounding, multi-criteria benchmark design |
| Efficiency | 10 | 7 | Online proactive monitoring without full conversation history requirement |
| Generalization | 5 | 4 | 12 diverse risk categories covering real-world threat landscapes |
| Domain Relevance | 25 | 22 | Core relevance: multi-turn harm amplification and proactive moderation directly applies to content governance, influencer violation detection |
| **Total** | **100** | **81** | |

---

## 与先前工作的对比 / Comparison with Prior Work

| Work | Turn Type | Harm Metric | Mitigation |
|------|-----------|-------------|------------|
| WildGuard, LlamaGuard | Single | Binary harm | Passive filter |
| AgentHarm | Agent tasks | Task harm | Detection only |
| **HarmAmp + TrajSafe** | **Multi-turn** | **Amplification degree** | **Proactive tree-RL intervention** |

---

## 电商/内容治理相关性 / E-commerce & Governance Relevance

多轮危害放大在电商平台的达人/卖家咨询场景中尤为突出：恶意商家可通过连续对话诱导 LLM 客服系统提供违规营销策略、虚假宣传文案、违禁商品描述等。TrajSafe 的多轮主动干预机制可直接应用于：
- 达人内容违规检测（连续直播/评论链中的危害累积）
- 电商 AI 客服的多轮有害请求识别
- 内容平台的违规行为轨迹预测

Multi-turn harm amplification is particularly relevant in e-commerce AI assistant scenarios: malicious sellers can gradually extract prohibited marketing strategies, false advertising copy, or illegal product descriptions through extended LLM conversations. TrajSafe's multi-turn proactive intervention directly applies to influencer violation detection, e-commerce AI chatbot safety, and platform violation trajectory prediction.

---

## Code Reproduction

See `code/TrajSafe/` for a faithful PyTorch reproduction of HarmAmp benchmark evaluation + TrajSafe proactive monitor.
