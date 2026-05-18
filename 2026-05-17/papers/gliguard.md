# GLiGuard: Schema-Conditioned Classification for LLM Safeguard

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | GLiGuard: Schema-Conditioned Classification for LLM Safeguard |
| **Authors** | Fastino AI team |
| **ArXiv** | https://arxiv.org/abs/2605.07982 |
| **Submitted** | 8 May 2026 |
| **Venue** | arXiv preprint |
| **Code** | https://github.com/fastino-ai/GLiGuard |
| **Model** | https://huggingface.co/fastino/gliguard-LLMGuardrails-300M |
| **Domain** | LLM safety, content moderation, guardrails, efficient inference |

---

## 方法概述 / Method Overview

### 故事弧线 / Story Arc

> **现有不足**: 当前 SOTA 护栏模型（如 LlamaGuard、WildGuard）依赖 7B–27B 参数的自回归解码器，将多标签分类重新表述为文本生成，导致高延迟、高成本，且多任务并行评估效率低下。  
> **我们的设计**: GLiGuard 是一个 0.3B 参数的双向编码器（改编自 GLiNER2），将安全任务定义和标签语义直接编码进输入 token 序列（schema conditioning），在单次前向传播中同时完成 prompt 安全性、response 安全性、拒绝检测和 14 个细粒度危害类别的评估。

### 技术细节 / Technical Details

**架构创新**:
- 基于 GLiNER2 双向编码器，非自回归解码器
- **Schema Conditioning**: 将任务名称和候选标签结构化后编码到输入序列中
- 单次前向传播同时打分所有安全任务
- 支持多任务同时评估：prompt safety、response safety、拒绝检测、14类危害分类

**性能指标**:
- 参数量: ~0.3B（对比基线 7B–27B）
- 吞吐量: 193 req/s（A100 动态批处理，P99 延迟 < 1s）
- 速度: 16× faster than autoregressive guardrails
- 准确率: 与 23–90× 更大的模型相当

---

## 创新性分析 / Innovation Analysis

| 维度 | 分析 |
|------|------|
| **参数效率** | 0.3B vs 7B–27B，在保持竞争力准确率的同时大幅降低部署成本 |
| **推理速度** | 双向编码器并行解码，16× 更低延迟，适合实时内容审核 |
| **任务统一** | 单模型同时处理 prompt safety + response safety + 拒绝检测 + 14类危害，减少运维复杂度 |
| **Schema设计** | 动态标签 schema 注入，无需针对新任务重新训练 |
| **vs 先前工作** | 相比 LlamaGuard-3（8B）、WildGuard（7B），速度提升 16× 且精度相当 |

---

## 关键指标 / Key Metrics

| 评测集 | 指标 | GLiGuard (0.3B) | LlamaGuard-3 (8B) | WildGuard (7B) |
|--------|------|-----------------|-------------------|----------------|
| Safety benchmarks (avg) | Accuracy | Competitive | Baseline | Baseline |
| Throughput (A100) | req/s | **193** | ~12 | ~12 |
| P99 Latency | ms | **<1000** | ~8000 | ~8000 |
| Model Size | B params | **0.3** | 8 | 7 |
| Speedup | × | **16×** | 1× | 1× |

---

## 评分 / Scoring

| 维度 | 满分 | 得分 | 理由 |
|------|------|------|------|
| Innovation | 30 | 22 | Schema conditioning 双向编码器用于多任务安全分类，工程创新显著 |
| Experimental SOTA delta | 15 | 10 | 精度相当但效率大幅提升，效率创新明确 |
| Experimental quality/ablations | 15 | 10 | 多基准对比，吞吐量测试充分 |
| Efficiency | 10 | 9 | 16× 速度提升是核心卖点，实测数据详尽 |
| Generalization | 5 | 3 | 仅测试英文场景，中文/多语言泛化性未知 |
| **Domain relevance** | **25** | **18** | LLM 内容审核护栏，适用于电商平台 LLM API 安全层，但非电商专用 |
| **Total** | **100** | **72** | 工程价值高，适合平台实时内容审核部署 |
