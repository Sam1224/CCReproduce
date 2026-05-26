# GLiGuard: Schema-Conditioned Classification for LLM Safeguard

## 基本信息 / Basic Info

| Field | Value |
|-------|-------|
| **Title** | GLiGuard: Schema-Conditioned Classification for LLM Safeguard |
| **arXiv** | [2605.07982](https://arxiv.org/abs/2605.07982) |
| **Authors** | Urchade Zaratiana, Mary Newhauser, George Hurn-Maloney, Ash Lewis |
| **Affiliations** | Fastino Labs |
| **Date** | 2026-05-08 |
| **Bucket** | STRONG |
| **Total** | **77 / 100** |
| **Code** | [github.com/fastino-ai/GLiGuard](https://github.com/fastino-ai/GLiGuard) ✅ |

---

## 故事弧 / Story Arc

> **问题:** LLM内容安全审核需要实时、多维度评估（提示安全、响应安全、越狱策略、14种细粒度危害类别），而基于解码器的大型guard模型推理延迟高、无法满足线上实时需求。
>
> **方案:** GLiGuard是一个0.3B参数的双向编码器（基于GLiNER2适配），通过将任务定义和标签语义编码为**结构化token schema**直接注入输入序列，实现单次非自回归前向推理完成所有安全维度的分类。
>
> **差异:** 不同于先前基于解码器的guard模型（Llama-Guard等），GLiGuard通过schema条件化完全消除自回归生成开销，体积缩小23-90倍的同时性能差距仅1.7 F1点。

---

## 方法概述 / Method Summary

GLiGuard在GLiNER2基础上适配，核心设计是**Schema-Conditioned Input Format**：

```
[schema_token] prompt_safety [SEP] response_safety [SEP] 
harm_category_1: violence [SEP] harm_category_2: sexual [SEP] ...
[input_text] {actual prompt/response content}
```

- **结构化schema注入:** 将所有分类维度的语义描述拼接为前缀token序列
- **单次前向推理:** 双向编码器一次性产出所有维度分类结果（无需多次调用）
- **多维度覆盖:** 同时评估提示安全性、响应安全性、拒绝检测、14种危害类别、越狱策略

**推理流程:**
```
Input: [Schema tokens] + [Prompt/Response text]
  ↓ Bidirectional Encoder (0.3B)
Output: [Multi-label classification for all safety dimensions]
```

---

## 创新性分析 / Innovation

1. **Schema条件化设计:** 将任务描述嵌入输入而非模型架构，使同一模型可灵活覆盖任意新增安全类别
2. **极致效率:** 单次非自回归推理，延迟比LLaMA-Guard类模型降低16倍
3. **双向编码器优势:** 完整上下文理解，尤其对长响应内容的安全判断更准确
4. **可行性:** 代码已开源，适合工业级低延迟安全审核部署

---

## 关键指标 / Key Metrics

| Task | Metric | GLiGuard (0.3B) | Best Baseline |
|------|--------|-----------------|---------------|
| Prompt safety (avg) | F1 | within -1.7 | Largest decoder guard |
| Response safety (avg) | F1 | 2nd best | Decoder guards (23-90× larger) |
| Latency | ms/sample | **16× faster** | Decoder-based guards |

---

## 评分详情 / Score Breakdown

| Dimension | Score | Max | Justification |
|-----------|-------|-----|---------------|
| Innovation | 23 | 30 | Schema条件化方案创新；将任务语义编码为token schema是新颖设计 |
| Experimental SOTA delta | 12 | 15 | 16倍速度提升；F1差距仅1.7点 |
| Experimental quality / ablations | 11 | 15 | 多基准评估；schema消融合理 |
| Efficiency | 9 | 10 | 0.3B模型，16倍延迟优势，显著工业价值 |
| Generalization | 3 | 5 | 适用多种LLM平台，但schema仍需定制 |
| Domain relevance (ecom + governance) | 19 | 25 | LLM内容安全直接用于电商平台内容审核和达人互动安全 |
| **Total** | **77** | **100** | — |

---

## 与本领域关联 / Domain Relevance

- **平台内容安全:** 电商平台LLM对话系统（客服、导购AI）需要实时安全审核
- **达人内容合规:** AI生成的达人营销内容需通过安全过滤确保合规
- **低延迟关键:** 电商场景毫秒级响应要求与GLiGuard的16倍速度优势高度匹配
