# GLiGuard: Schema-Conditioned Classification for LLM Safeguard

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **标题** | GLiGuard: Schema-Conditioned Classification for LLM Safeguard |
| **作者** | Urchade Zaratiana, Mary Newhauser, George Hurn-Maloney, Ash Lewis |
| **机构** | Fastino Labs |
| **arXiv ID** | [2605.07982](https://arxiv.org/abs/2605.07982) |
| **提交日期** | May 13, 2026 |
| **代码** | [github.com/fastino-ai/GLiGuard](https://github.com/fastino-ai/GLiGuard) + HuggingFace: `fastino/gliguard-LLMGuardrails-300M` |
| **Bucket** | STRONG |

---

## 方法概述 / Method Overview

**EN:** State-of-the-art LLM guardrail models rely on autoregressive decoders with 7B–27B parameters, treating what is fundamentally a multi-aspect classification problem (is this prompt safe? is this response safe? what harm category? is there jailbreak?) as sequential text generation. This incurs prohibitive latency (40–200ms per call) and scales poorly.

GLiGuard adapts **GLiNER2** (a bidirectional encoder) into a **0.3B-parameter schema-conditioned classifier** for LLM content moderation. The key insight is to frame guardrailing as multi-label classification where the schema defines the output space: given a user prompt + model response pair, GLiGuard simultaneously evaluates (1) prompt safety, (2) response safety, (3) fine-grained harm category, and (4) jailbreak strategy — all in a single forward pass. The "schema conditioning" means harm categories and policy definitions are embedded directly into the encoder's attention through a schema-token prefix, enabling zero-shot generalization to new policy namespaces without retraining.

**ZH:** 现有最强守门模型（如 LlamaGuard-3 27B）用自回归解码器处理本质上是分类问题的安全判断，带来极高推理延迟。GLiGuard 将 GLiNER2（双向编码器）改造为 0.3B 参数的**模式条件化多任务分类器**，一次前向传播同时评估：提示安全性、响应安全性、细粒度危害类别及越狱策略，并通过"模式前缀嵌入"支持零样本泛化到新政策命名空间。

---

## 故事主线 / Story Arc

> **现有方法的不足:** 7B–27B 的自回归守门模型延迟高、吞吐低，无法支撑电商/内容平台每秒数万次的实时审核请求。
>
> **我们的解决方案:** 将安全审核重新建模为多方面分类任务，用 0.3B 双向编码器替代大型解码器，在保持准确率的同时实现 16× 吞吐提升和 17× 延迟降低。

---

## 创新性分析 / Innovation Analysis

1. **分类视角：** 首次系统论证守门任务是分类而非生成，正式化了"错误的问题框架"问题。
2. **模式条件化：** schema token prefix 允许在不重训练的情况下扩展新危害类别，高度工程化。
3. **单次前向传播多任务：** 提示安全、响应安全、危害分类、越狱检测四合一，极大降低系统复杂度。
4. **vs. 先前工作：** LlamaGuard 系列均为解码器方案，GLiGuard 是首个在 9 个基准上与 7B-27B 模型对齐的亚 1B 编码器守门模型。

---

## 关键指标 / Key Metrics

| Benchmark | Metric | GLiGuard (0.3B) | LlamaGuard3-7B | LlamaGuard3-27B |
|-----------|--------|-----------------|-----------------|-----------------|
| 9 safety benchmarks (avg) | F1 | ~88% | ~87% | ~89% |
| Throughput | Req/s | 16× higher | 1× | — |
| Latency | ms | 17× lower | 1× | — |
| Model size | Params | 0.3B | 7B | 27B |

> **Size ratio:** GLiGuard is 23–90× smaller than comparable decoder guards while matching or exceeding F1 on 9 established safety benchmarks.

---

## 评分 / Scoring

| 维度 | 分数 | 理由 |
|------|------|------|
| Innovation | 20/30 | 巧妙的框架转换，但基于已有 GLiNER2 架构改造 |
| Experimental SOTA delta | 14/15 | 以 0.3B 媲美 27B，Δ≈0 准确率、16× 效率提升 |
| Experimental quality | 13/15 | 9 个基准，多方面评估 |
| Efficiency | 10/10 | 16× 吞吐、17× 延迟，生产级验证 |
| Generalization | 4/5 | 跨多个安全维度泛化 |
| Domain relevance | 20/25 | 直接对应平台实时内容安全审核，生产可用 |
| **Total** | **81/100** | |

**Code available:** [github.com/fastino-ai/GLiGuard](https://github.com/fastino-ai/GLiGuard) — 实际实现通过 GLiNER2 库分发，包含完整推理接口和 schema 定义格式。
