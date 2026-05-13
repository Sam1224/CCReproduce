# GLiGuard: Schema-Conditioned Classification for LLM Safeguard

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| 标题 | GLiGuard: Schema-Conditioned Classification for LLM Safeguard |
| 作者 | Urchade Zaratiana, Mary Newhauser, George Hurn-Maloney, Ash Lewis |
| 机构 | Fastino AI (GLiNER2 团队) |
| arXiv | https://arxiv.org/abs/2605.07982 |
| 提交日期 | 2026-05-08 (arXiv 列表：2026-05-11) |
| 领域标签 | 内容审核 · LLM 安全 · 高效编码器 · Schema 分类 · 多维度安全评估 |
| 桶类别 | STRONG |
| 综合评分 | **86 / 100** ⭐ |
| 代码复现 | `code/GLiGuard/` |

---

## 方法概述 (中文)

大型语言模型输出安全审核（LLM Safeguard/Guardrail）目前主流方案均采用 7B–27B 参数的自回归解码器，将本质上是分类问题的安全评估转化为顺序文本生成任务，造成高延迟和低吞吐的双重困境。

**GLiGuard 的核心思路：**
1. **架构选择**：适配 GLiNER2 的双向 Transformer 编码器（~0.3B 参数），而非自回归解码器。双向注意力使模型能同时感知上下文，无需逐 token 生成。
2. **Schema Token 输入**：将任务定义（prompt safety / response safety / refusal detection）、14 个细粒度危害类别（暴力、自伤、色情、歧视等）和 11 种越狱策略，作为结构化 token schema 拼接到输入序列头部，使模型在一次前向传播中即完成全部维度评估。
3. **多维度非自回归推理**：单次 forward pass 输出所有安全维度的分类决策，不依赖 beam search 或 chain-of-thought 生成。
4. **训练**：在现有 9 个安全基准的数据上进行监督微调，label 为各维度标注。

---

## 故事线 (Story Arc)

> **现状不足：** 当前最强 guardrail 模型（如 LlamaGuard 系列 7B–27B）将安全分类重新表述为"生成文本判断"，这在推理延迟上付出巨大代价——对于需要实时内容过滤的平台（直播、内容审核流水线、API 网关），每秒处理数百万请求时 7B 模型的高延迟是不可接受的。
>
> **我们的解法：** 将安全分类还原为判别式任务。将安全维度的语义（任务定义 + 类别描述）编码为 schema token 前缀，使 0.3B 双向编码器能在单次前向传播中完成多维安全判断，在保持 F1 接近 7B–27B 基线的同时，实现 16× 吞吐提升和 17× 延迟降低。

---

## 创新点分析

| 维度 | 描述 |
|------|------|
| 核心创新 | 将 GLiNER2 的 Schema-driven 信息抽取范式迁移到 LLM 安全审核，实现多维度判别式单次推理 |
| vs. 先前工作 | 先前 guardrail 均为生成式（自回归），无法高效多维并发；GLiGuard 为首个将 schema 信息编码到输入前缀实现多任务判别式安全审核的工作 |
| 可行性 | F1 与 7B–27B 模型差距 ≤ 1.7 分，有充分实验证明可实际部署 |
| 局限 | 无法输出链式推理解释；对新型越狱策略需重新加入 schema 并微调 |
| 效率优势 | 0.3B vs 7–27B → 23–90× 参数压缩，16× 吞吐，17× 延迟 |

---

## 关键指标

| 数据集/基准 | 指标 | GLiGuard (0.3B) | 最强基线 |
|------------|------|-----------------|---------|
| 9 safety benchmarks avg | Prompt Safety F1 | competitive | 7B–27B models |
| 9 safety benchmarks avg | Response Safety F1 | 2nd best among open guards | 7B–27B models |
| 推理效率 | Throughput | 16× 高于 7B 基线 | 7B autoregressive |
| 推理效率 | Latency | 17× 低于 7B 基线 | 7B autoregressive |
| 模型大小 | Parameters | 0.3B | 7B–27B |

---

## 评分分解

| 维度 | 得分 | 满分 | 说明 |
|------|------|------|------|
| 创新性 | 25 | 30 | 将判别式 Schema 范式引入安全审核，核心迁移思路清晰且有效；绝对原创性略受 GLiNER2 基础框架限制 |
| 实验 SOTA Delta | 13 | 15 | 9 个基准全面评测，效率指标突出，F1 差距 ≤ 1.7 分非常有竞争力 |
| 实验质量/消融 | 12 | 15 | 9 基准对比、效率消融清晰；不足：缺乏越狱策略的细粒度消融 |
| 效率 | 10 | 10 | 16×/17× 吞吐/延迟提升，参数压缩 23–90×，满分 |
| 泛化性 | 4 | 5 | 覆盖 14 危害类别 + 11 越狱策略，泛化性强；但未测中文/多语言场景 |
| 领域相关性 | 22 | 25 | 直接适用于内容审核平台 API 网关、直播流水线实时安全检测 |
| **Total** | **86** | **100** | — |

---

## 代码与数据

- 论文代码：未在 arXiv 页面列出（截至发现时）
- 本仓库复现：`code/GLiGuard/`
- 基础依赖：GLiNER2 (github.com/fastino-ai/GLiNER2)、transformers、PyTorch

---

## 方法图示说明

```
Input Sequence:
[TASK: prompt_safety] [LABEL: violence] [LABEL: self_harm] ... [LABEL: jailbreak_persona] 
[SEP] <User Prompt Text> [SEP]

      ↓  Bidirectional Transformer Encoder (0.3B)  ↓

[CLS token representation] → Linear classifiers → {safe, unsafe} per dimension
All dimensions decided in ONE forward pass (non-autoregressive)
```
