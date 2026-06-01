# GLiGuard: Schema-Conditioned Classification for LLM Safeguard

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **标题** | GLiGuard: Schema-Conditioned Classification for LLM Safeguard |
| **arXiv ID** | 2605.07982 |
| **提交日期** | 2026-05-08 |
| **作者** | Urchade Zaratiana, Mary Newhauser, George Hurn-Maloney, Ash Lewis |
| **机构** | Fastino AI |
| **论文链接** | https://arxiv.org/abs/2605.07982 |
| **官方代码** | https://github.com/fastino-ai/GLiGuard |
| **HuggingFace** | https://huggingface.co/fastino/gliguard-LLMGuardrails-300M |
| **桶** | WEAK→STRONG |
| **Total** | **69** |

---

## 方法概述 / Method

**故事弧（Story Arc）：**
> 当前主流 LLM 防护（Guardrail）模型使用 7B-27B 自回归解码器，将本质是分类任务的安全判定重新表述为文本生成——这导致高延迟（17x 对比本文）和低扩展性（多维度同步评估代价极高）。GLiGuard 将安全防护**重新定义为结构化分类**：用 0.3B 双向编码器（基于 GLiNER2），将任务定义和标签语义编码为输入序列的"schema token"，单次前向传播即可完成 14 个伤害类别 + 多个安全维度的**同时**分类。

**架构：**
```
输入: "User Prompt / Model Response" + Schema Tokens (task def + label semantics)
         ↓
[Bidirectional Encoder, 0.3B, GLiNER2 架构]
  ← schema 信息直接融入 input sequence →
         ↓
Single Forward Pass
  → prompt safety score
  → response safety score
  → refusal detection
  → 14x fine-grained harm category scores
```

**与前工作差异：**
- LlamaGuard / ShieldGemma：7B-27B 自回归，单次只判断一个维度
- GLiGuard：0.3B 双向编码器，单次并行判断所有维度

---

## 关键指标 / Key Metrics

| 指标 | GLiGuard (0.3B) | 最佳对比（7B-27B）|
|------|-----------------|-----------------|
| 分类准确率 | comparable | — |
| 推理速度 | **16x faster** | baseline |
| 推理延迟 | **17x lower** | baseline |
| 模型尺寸 | **23-90x smaller** | 7B-27B |

---

## 评分 / Scoring

| 维度 | 子分 | 说明 |
|------|------|------|
| Innovation (max 30) | 20 | Schema-conditioned 双向编码器用于安全防护；GLiNER2 的创意迁移 |
| SOTA Δ (max 15) | 12 | 16x 速度提升，精度持平 7-27B 模型 |
| Experimental Quality (max 15) | 10 | 14 类 harm，多安全维度 ablation |
| Efficiency (max 10) | 9 | **核心贡献**：0.3B 替代 7-27B，生产可用 |
| Generalization (max 5) | 3 | LLM 内容安全专向，可扩展到平台审核 |
| Domain Relevance (max 25) | 15 | LLM 安全防护→可迁移到平台内容审核 pipeline |
| **Total** | **69** | — |

---

## 创新性分析

1. **分类重新定义**：将自回归 guardrail 问题还原为判别分类——这一"设计选择"带来了 16x 速度提升和 90x 模型压缩，体现了工程洞察。
2. **Schema Token 设计**：任务定义和标签语义以结构化 token 编码入输入序列，使同一模型无需重训即可评估任意新定义的安全维度。
3. **实用性极强**：0.3B 模型 + 开源 + 支持多维度并行，适合集成到电商平台的实时内容安全 API。

---

## 电商 / 达人治理落地思路

- 作为电商平台 LLM 服务的安全 guardrail 层
- 可将"广告法违规 / 虚假宣传 / 违禁品"定义为自定义 schema，无需重训
- 低延迟使其适合嵌入内容发布流程的实时审核 pipeline
