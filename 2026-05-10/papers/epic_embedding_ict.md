## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **标题** | Embedding-based In-Context Prompt Training for Enhancing LLMs as Text Encoders |
| **arXiv ID** | [2605.01372](https://arxiv.org/abs/2605.01372) |
| **提交日期** | 2026-05-02 |
| **作者** | Ailiang Lin, Zhuoyun Li, Keyu Mao, Kotaro Funakoshi, Manabu Okumura |
| **机构** | Tokyo Institute of Technology |
| **领域** | Text Embedding · In-Context Learning · LLM Encoder · MTEB |
| **Bucket** | WEAK |

---

## 方法概述 / Method Summary

EPIC（Embedding-based In-Context Prompt Training）解决了 LLM 用于嵌入生成时的 ICL（In-Context Learning）高 token 开销问题：

**核心思想：** 传统 ICL 通过拼接文本示例提升表示质量，但导致序列长度大幅增加。EPIC 用连续嵌入向量替代离散文本示例，作为 In-Context Prompt 注入模型。

**训练策略：**
1. 将 few-shot 演示的文本表示替换为对应嵌入向量（continuous prompt）
2. 对比学习：促使 LLM 对语义相关文本对对齐
3. 同时要求模型将演示嵌入作为 In-Context 提示进行语义解读

**效果：** 在完全基于公开检索数据训练的模型中，在 MTEB 基准上达到新 SOTA，超越 frontier 模型。

### Story Arc
> **ICL 显著提升 LLM 嵌入质量，但离散文本示例导致序列长度爆炸** → EPIC 用连续嵌入向量替代离散示例，既保留 ICL 的语义对齐效果，又大幅降低训练和推理时的 token 开销，在 MTEB 上达到最优。

---

## 关键指标 / Key Metrics

| 基准 | 指标 | EPIC | Previous SOTA |
|------|------|------|--------------|
| MTEB (publicly available data) | Average Score | **SOTA** | — |

---

## 评分 / Scoring

| 维度 | 得分 | 说明 |
|------|------|------|
| Innovation | 17/30 | 用嵌入替代离散 ICL 示例的思路新颖 |
| SOTA Delta | 11/15 | MTEB 公开数据 SOTA，但限定在"公开数据"条件下 |
| Exp Quality / Ablations | 10/15 | MTEB 全任务评估 |
| Efficiency | 8/10 | 大幅降低 token 开销 |
| Generalization | 4/5 | MTEB 覆盖多种下游任务 |
| Domain Relevance | 10/25 | 通用文本嵌入，可用于电商商品语义检索 |
| **总分** | **60/100** | Feishu 推送 |
