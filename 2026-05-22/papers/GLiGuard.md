# GLiGuard: Schema-Conditioned Classification for LLM Safeguard

## 基本信息 / Basic Info

| 字段 | 值 |
|------|-----|
| **Title** | GLiGuard: Schema-Conditioned Classification for LLM Safeguard |
| **arXiv** | https://arxiv.org/abs/2605.07982 |
| **Authors** | Urchade Zaratiana, Mary Newhauser, George Hurn-Maloney, Ash Lewis |
| **Affiliation** | Fastino Labs |
| **Submitted** | May 8, 2026 |
| **Domain Tags** | `content-moderation` `LLM-safeguard` `schema-classification` `efficiency` `content-governance` |
| **Code** | https://github.com/fastino-ai/GLiGuard |
| **Model** | https://huggingface.co/fastino/gliguard-LLMGuardrails-300M |
| **Bucket** | STRONG |
| **Total** | **82** |

---

## 方法概述 / Method Overview

**中文：**
GLiGuard 是一个基于 GLiNER2 架构的 0.3B 参数双向编码器模型，专为大语言模型内容安全审核设计。现有的 guardrail 模型（如 LlamaGuard、ShieldGemma）依赖 7B–27B 自回归解码器，导致高延迟和多任务扩展困难。GLiGuard 将安全分类重新定义为结构化信息抽取问题：将任务名称和候选标签作为结构化 token schema 嵌入输入序列，通过单次双向前向传播同时评估提示安全性、响应安全性、越狱检测和细粒度危害类别。

**English:**
GLiGuard is a 0.3B-parameter bidirectional encoder adapted from GLiNER2 for LLM content moderation. It reframes safety as structured classification: task names and candidate labels are encoded directly into the input sequence as a structured token schema, enabling simultaneous evaluation of prompt safety, response safety, jailbreak detection, and fine-grained harm categories in a single forward pass — without changing model heads or prompt templates.

---

## 故事线 / Story Arc

现有 guardrail 模型（7B–27B 解码器）推理速度慢、延迟高，难以实时部署 →  
我们把多任务安全审核统一建模为 schema-conditioned 分类 →  
0.3B 双向编码器在 9 个安全基准上匹配或超越 23–90× 更大的模型 →  
在工业级吞吐要求下实现 16× 吞吐提升 + 17× 延迟降低。

---

## 创新性分析 / Innovation

- **核心创新**：将 LLM 内容安全审核统一为 schema-conditioned 多标签分类，无需为每个安全维度维护独立模型或 prompt 模板。
- **架构创新**：基于 GLiNER2 双向编码器而非自回归解码器，单次前向传播完成所有审核维度。
- **与先前工作的差异**：LlamaGuard/ShieldGemma 等使用大型解码器 LLM，需要生成式推理；GLiGuard 的编码器方式在准确率相当的情况下将计算开销降低了 1–2 个数量级。
- **创新可行性**：GLiNER2 成功先例 + 实测 9 个基准的数据 → 可信度高。

---

## 关键指标 / Key Metrics

| Benchmark Suite | Metric | GLiGuard (0.3B) | Baseline (7B–27B) |
|----------------|--------|----------------|-------------------|
| 9 established safety benchmarks | F1 | Competitive | Reference |
| Throughput | tokens/s | **16× 提升** | 1× |
| Latency | ms/request | **17× 降低** | 1× |
| Model size | parameters | **0.3B** | 7B–27B |

---

## 评分 / Scoring

| 维度 | 分 / 满分 | 理由 |
|------|----------|------|
| 创新性 | 22 / 30 | 将多任务安全审核重建模为 schema 分类是明确创新，基于 GLiNER2 的工程实现高效；但仍属分类任务范式 |
| 实验指标 | 13 / 15 | 9 个基准上与 7B-27B 模型竞争、16×/17× 效率提升，证据充分 |
| 实验质量 | 13 / 15 | 9 个多维度基准，系统性消融；缺少在线 A/B 测试 |
| 效率 | 10 / 10 | 23–90× 参数缩减、16× 吞吐、17× 延迟，工业部署友好 |
| 泛化性 | 4 / 5 | 9 个基准跨多场景 |
| 相关性 | 20 / 25 | 直接适用于电商平台违规检测（虚假宣传、违禁词、不当内容）和达人内容治理管线 |
| **Total** | **82** | |

---

## 代码说明 / Code Note

官方代码已开源，包含完整 Python 推理接口（GLiNER2 库）：
- GitHub: https://github.com/fastino-ai/GLiGuard
- HuggingFace: https://huggingface.co/fastino/gliguard-LLMGuardrails-300M
- 许可证: Apache 2.0

官方实现完整，无需额外复现。参见 `code/GLiGuard/` 中的使用示例封装。
