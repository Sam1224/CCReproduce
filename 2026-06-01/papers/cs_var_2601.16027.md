# Deja Vu in Plots: Leveraging Cross-Session Evidence with Retrieval-Augmented LLMs for Live Streaming Risk Assessment

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | Deja Vu in Plots: Leveraging Cross-Session Evidence with Retrieval-Augmented LLMs for Live Streaming Risk Assessment |
| **Authors** | Yiran Qiao, Xiang Ao, Jing Chen, Yang Liu, Qiwei Zhong, Qing He |
| **Affiliation** | Chinese Academy of Sciences (Institute of Computing Technology) |
| **Venue** | arXiv preprint (submitted January 22, 2026) |
| **arXiv** | https://arxiv.org/abs/2601.16027 |
| **Submitted** | 2026-01-22 |
| **Code** | `code/CS-VAR/` |

---

## 方法概述 / Method Summary

直播平台的欺诈、协同恶意行为等风险往往跨会话积累、复现，单次会话信号不足以支撑准确判断。2025 年 1 月至 8 月间，电商直播平台已有逾 5 万名主播被处罚，但现有方法局限于单场会话内分析。

本文提出 **CS-VAR (Cross-Session Evidence-Aware Retrieval-Augmented Detector)**，核心思想：

1. **跨会话检索（Cross-Session Retrieval）**：给定当前直播会话，从历史会话数据库中检索与该主播行为模式相似的历史会话（包含完整的行为序列）。
2. **LLM 跨会话推理（LLM Teacher Reasoning）**：以 LLM 作为"教师"，对当前会话与检索到的历史证据进行联合推理，识别跨会话的重复恶意模式，生成"局部→全局"的结构化风险判断。
3. **知识蒸馏（Knowledge Distillation）**：LLM 的推理输出用于指导一个轻量级领域特定模型（学生）的训练，使其在不调用 LLM 的前提下完成实时风险推理，满足线上部署的延迟要求。
4. **可解释输出**：小模型输出局部化的、可定位的风险信号，可直接支持人工审核员的决策。

**故事弧线：** 传统方法只看单次会话，而恶意主播惯用"换汤不换药"的手法跨场复现 → CS-VAR 以 RAG-LLM 挖掘跨会话复发模式，再通过蒸馏实现实时部署，在大规模工业数据集上达到 SOTA。

---

## 创新性分析 / Innovation Analysis

- **新颖点1 — 跨会话证据检索**：首次将 RAG 范式引入直播风险检测，以历史会话作为知识库，解决了跨场景模式识别问题。
- **新颖点2 — LLM 教师 + 小模型学生蒸馏**：训练阶段 LLM 推理丰富的跨会话上下文；推理阶段轻量学生模型独立运行，实现"训练重，推理轻"的工业可用设计。
- **新颖点3 — 可解释局部化信号**：输出不只是 risk score，还有可定位的风险事件位置，便于运营人员复核。
- **与前作区别**：此前直播风险检测（如 Live or Lie, ArXiv:2602.03520）侧重单场会话多实例学习；CS-VAR 跨越多场会话，利用 LLM 的"全局鸟瞰"发现重复恶意手法，系首次显式建模跨会话证据。

---

## 关键指标 / Key Metrics

| Dataset | Metric | This Work | Baseline |
|---------|--------|-----------|---------|
| Large-scale Industrial Dataset (内部) | Offline SOTA | ✓ (stated SOTA) | — |
| Online Validation | Business KPIs | Positive (stated) | — |

> 详细数值在摘要级别未公开；完整结果见论文表格。

---

## 评分 / Score

| Dimension | Score | Max | Justification |
|-----------|-------|-----|---------------|
| Innovation | 24 | 30 | 跨会话 RAG + LLM→小模型蒸馏的组合创新，在直播风险场景下属首创 |
| Experimental SOTA Delta | 11 | 15 | 工业级数据集 SOTA + 线上验证，但具体数值摘要中未完全披露 |
| Experimental Quality / Ablations | 11 | 15 | 线上 + 线下双重验证，工业场景真实性高 |
| Efficiency | 8 | 10 | 蒸馏后推理不需 LLM，满足实时需求 |
| Generalization | 4 | 5 | 跨会话模式识别具有通用性，但数据集内部 |
| Domain Relevance (ecom + governance) | 23 | 25 | 电商直播风险评估，达人治理核心场景 |
| **Total** | **81** | **100** | |
