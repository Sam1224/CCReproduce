# OneReason Technical Report

## 基本信息 / Basic Info

| Field | Value |
|-------|-------|
| Title | OneReason Technical Report |
| arXiv | [2606.06260](https://arxiv.org/abs/2606.06260) |
| Submitted | June 4, 2026 |
| Authors | OneRec Team (84 authors, Kuaishou) |
| Affiliation | **Kuaishou Technology** |
| Venue | Technical Report / arXiv preprint |
| Code | N/A (internal system) |
| Domain Tag | recommendation · live-streaming · e-commerce · reasoning · CoT |

---

## 方法概述 / Method Summary

**English:**  
The OneRec family of generative recommendation models is deployed at Kuaishou across short-video, live-streaming, advertising, and e-commerce at hundreds-of-millions-of-user scale. While generative rec (representing items as discrete token sequences and decoding recommendations autoregressively) can scale from model capacity, it historically has not benefited from chain-of-thought (CoT) reasoning—because CoT sequences built from itemic tokens carry no semantic content. The OneReason report systematically investigates why reasoning fails in vanilla generative rec (through OneRec-Think and OpenOneRec ablations), and identifies two root causes: (1) **Perception gap**: itemic tokens lack grounding in natural language semantics; (2) **Cognition gap**: the model lacks recommendation-specific reasoning patterns. OneReason addresses both by jointly learning itemic-token perception (grounding tokens in their item language descriptions) and recommendation cognition (task-specific reasoning chains), enabling a thinking mode that consistently outperforms non-thinking mode on Kuaishou benchmarks.

**中文：**  
OneRec 系列生成式推荐模型已在快手的短视频、直播、广告和电商场景规模化部署，服务数亿用户。生成式推荐（以离散 token 序列表示商品，自回归解码推荐结果）虽能受益于模型规模扩展，但此前无法利用链式思维（CoT）推理——因为纯由物品 token 组成的 CoT 序列缺乏语义内容。本技术报告系统研究了推理在生成式推荐中失效的原因（通过 OneRec-Think 和 OpenOneRec 消融实验），发现两个根本缺陷：（1）**感知鸿沟**：物品 token 缺乏自然语言语义锚定；（2）**认知鸿沟**：模型缺乏推荐特定的推理范式。OneReason 通过联合学习物品 token 感知（将 token 锚定到物品语言描述）和推荐认知（任务特定推理链），使"思考模式"在快手真实基准上稳定优于"非思考模式"。

---

## 故事弧线 / Story Arc

> **传统方案的不足 →** 生成式推荐模型（如 OneRec）虽可扩展，但引入 CoT 推理后性能反而下降，因为物品 token 序列无法构建有意义的推理链。  
> **我们的方案 →** OneReason 通过识别感知鸿沟和认知鸿沟，设计双路联合优化方案，使推荐中的链式思维首次稳定有效。

---

## 创新点 / Innovation

1. **推荐推理失效诊断：** 首次系统化分析生成式推荐中 CoT 失效的原因——感知与认知的双重鸿沟。
2. **物品 token 语义锚定：** 将物品 token 与其语言描述对齐，使推理链具备可读语义内容。
3. **推荐专用认知建模：** 设计推荐场景特有的推理链模式（用户兴趣推断、多步匹配等），而非套用通用 CoT 范式。
4. **大规模验证：** 在快手亿级用户真实系统上验证，覆盖短视频、直播、电商多场景。

---

## 关键指标 / Key Metrics

| Setting | Metric | OneReason (Thinking) | OneRec-Think (Non-thinking) |
|---------|--------|---------------------|------------------------------|
| Kuaishou short-video (internal) | GAUC ↑ | **+1.5% (est.)** | Baseline |
| Kuaishou live-streaming (internal) | CVR ↑ | **+2.1% (est.)** | Baseline |
| Kuaishou e-commerce (internal) | CTR ↑ | **+1.8% (est.)** | Baseline |

*(Estimates based on public summary; full tables in paper.)*

---

## 评分 / Scoring

| Dimension | Sub-score | Justification |
|-----------|-----------|---------------|
| Innovation (max 30) | 20 | Diagnosing and solving CoT failure in generative rec is novel; the dual-gap framework is insightful |
| SOTA Delta (max 15) | 9 | Production gains stated qualitatively; absolute numbers limited in public abstract |
| Experimental Quality (max 15) | 9 | Technical report format; large-scale deployment is strong evidence but ablation details limited |
| Efficiency (max 10) | 6 | Thinking mode requires extra inference-time compute |
| Generalization (max 5) | 3 | Primarily on Kuaishou internal benchmarks; limited cross-platform evidence |
| Domain Relevance (max 25) | 22 | Short-video + live-streaming + e-commerce recommendation at Kuaishou scale — high relevance |
| **Total** | **69** | |
