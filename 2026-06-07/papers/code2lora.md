## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | Code2LoRA: Hypernetwork-Generated Adapters for Code Language Models under Software Evolution |
| **Authors** | Liliana Hotsko, Yinxi Li, Yuntian Deng, Pengyu Nie |
| **Affiliations** | University of Waterloo |
| **ArXiv ID** | [2606.06492](https://arxiv.org/abs/2606.06492) |
| **Submitted** | 2026-06-04 (indexed 2026-06-07 GMT+8) |
| **Categories** | cs.SE, cs.LG |
| **Code** | [anonymous.4open.science/r/code2lora-6857](https://anonymous.4open.science/r/code2lora-6857) |
| **Models** | [huggingface.co/code2lora](https://huggingface.co/code2lora) |
| **Bucket** | WEAK |
| **Total** | **72 / 100** |

---

## Score Breakdown

| Dimension | Score | Max | Justification |
|-----------|-------|-----|---------------|
| Innovation | 26 | 30 | Extending hypernetwork-generated LoRA to full repository conditioning + GRU-based evolution-aware adapter update is a genuine extension with clear motivation |
| Experimental SOTA delta | 12 | 15 | Static: Cross-repo EM=63.8%, In-repo EM=66.2%; Evo: +5.2pp over shared LoRA |
| Experimental quality / ablations | 13 | 15 | RepoPeftBench (604 Python repos, 40K train/12K test); two tracks; ablation over hypernetwork components |
| Efficiency | 8 | 10 | Zero inference-time token overhead; GRU update is lightweight; adapters generated in one pass |
| Generalization | 4 | 5 | Applicable to any code LM; repo-level conditioning is domain-agnostic |
| Domain relevance | 9 | 25 | Primarily a code engineering tool; indirect relevance as infrastructure for LLM-powered content governance automation |
| **Total** | **72** | **100** | |

---

## 方法概述 / Method Overview

### 问题背景（故事弧）
代码语言模型需要仓库级上下文来解析 import、API 调用和项目约定，但现有方法要么将仓库内容作为长输入（token 成本高昂），要么为每个仓库单独微调（计算成本高且对演化中的代码库脆弱）。

**X is insufficient → we design Y to solve it：**
> 现有仓库级适配方法在规模化部署时面临高推理成本或高训练成本的二选一困境。Code2LoRA 用超网络将整个仓库编码为固定向量，一次性生成该仓库专属的 LoRA 适配器，推理时无额外 token 开销。针对演化中的仓库，Code2LoRA-Evo 用 GRU 累积 commit diff 的隐状态，增量更新 adapter，将更新成本从"重训"降为"轻量增量推理"。

### 核心方法

1. **Code2LoRA-Static**：
   - 仓库编码器（Transformer）将所有文件内容编码为固定向量
   - 超网络（Hypernetwork）将仓库向量映射为 LoRA 权重矩阵
   - 一次性生成仓库专属 LoRA，注入基础代码 LM

2. **Code2LoRA-Evo**：
   - GRU 隐状态保存仓库历史表示
   - 每个 commit diff 触发一次 GRU 更新，生成新的 adapter
   - 避免静态 LoRA 随代码漂移失效

3. **RepoPeftBench**：604 个 Python 仓库，包含静态轨道（40K train/12K test）和演化轨道，评测指标为 Exact Match（EM）。

### English Summary

Code2LoRA encodes an entire repository into a fixed embedding and uses a hypernetwork to generate repository-specific LoRA adapters in a single pass, injecting repo knowledge into code LM weights with zero inference-time token overhead. For evolving codebases, Code2LoRA-Evo maintains a GRU hidden state updated by commit diffs, refreshing adapters incrementally. The authors also introduce RepoPeftBench (604 Python repos) with two evaluation tracks. Results show meaningful gains over shared LoRA baselines, with the evolution-aware variant providing an additional +5.2pp EM on the cross-repo track.

---

## 关键指标 / Key Metrics

| Track | Metric | Value | vs. Baseline |
|-------|--------|-------|-------------|
| Static, Cross-repo | EM | **63.8%** | — |
| Static, In-repo | EM | **66.2%** | — |
| Evo, Cross-repo | EM | **60.3%** | **+5.2pp** vs. shared LoRA |
