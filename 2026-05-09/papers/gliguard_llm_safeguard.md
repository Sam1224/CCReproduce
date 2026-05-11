# GLiGuard: Schema-Conditioned Classification for LLM Safeguard

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| 标题 | GLiGuard: Schema-Conditioned Classification for LLM Safeguard |
| 作者 | (GLiNER2 团队衍生, 作者信息部分可知) |
| 机构 | 未完全披露（基于 GLiNER2 架构） |
| arXiv | https://arxiv.org/abs/2605.07982 |
| 提交日期 | 2026-05-08 |
| 领域标签 | 内容审核 · LLM安全 · 模型蒸馏 · 轻量化 · 多任务分类 |
| 桶类别 | STRONG |
| 综合评分 | **83 / 100** |

---

## 方法概述 (中文)

现有 LLM 安全护栏（Guardrail）模型依赖大规模自回归解码器（7B–27B 参数），推理延迟高、吞吐量低，无法满足高并发线上内容审核场景的实时需求。单一模型通常只针对特定安全任务（如 prompt 安全或 response 安全），无法统一处理多维安全需求。

**GLiGuard** 将 LLM 护栏重新定义为 **schema-conditioned（模式条件化）多任务分类问题**：给定一个输入 schema（指定所需审核任务），模型在单次非自回归前向传播中同时完成以下评估：
1. **Prompt 安全性**（Prompt Safety）
2. **Response 安全性**（Response Safety）
3. **拒绝检测**（Refusal Detection）
4. **14 个细粒度有害类别**（如性暴力、仇恨、自伤等）
5. **11 种越狱策略检测**（Jailbreak Strategies）

架构基于 **GLiNER2 双向编码器**（0.3B 参数），通过 schema 注入实现多任务动态切换，无需针对每个任务分别训练独立模型。训练数据由高质量安全标注数据集构建，含 schema-task 对齐蒸馏步骤。

---

## 故事线 (Story Arc)

> **现状不足：** 当前 SOTA 护栏模型（如 LlamaGuard、ShieldLM 等）依赖 7B–27B 自回归解码器，推理延迟高（单次 100ms+），无法满足电商平台、内容平台每秒数万次审核请求的实时性要求；且每种安全任务需要分别调用不同模型，系统复杂度高。
>
> **我们的解法：** 利用轻量双向编码器（0.3B）+ schema 条件化设计，将所有安全任务统一为单次前向传播的多标签分类，实现与大模型护栏相当的 F1 分数，同时吞吐量提升 16 倍、延迟降低 17 倍——GLiGuard。

---

## 创新点分析

| 维度 | 描述 |
|------|------|
| 核心创新 | 将护栏重定义为 schema-conditioned 多任务分类；单次非自回归前向传播完成所有安全任务 |
| vs. 先前工作 | LlamaGuard/ShieldLM 等依赖 7B–27B 自回归解码；GLiGuard 用 0.3B 双向编码器替代，23–90× 更小 |
| 效率创新 | 16× 高吞吐、17× 低延迟；适合高并发内容审核流水线 |
| 可行性 | 基于成熟 GLiNER2 架构，工程可落地性强 |
| 局限 | 双向编码器在长上下文推理任务上可能不如生成式解码；14 类有害分类的召回率在极罕见类别上仍有改善空间 |

---

## 关键指标

| 数据集 / 任务 | 指标 | GLiGuard (0.3B) | 对比基线 |
|--------------|------|-----------------|---------|
| WildGuard / PromptBench | F1 (Prompt Safety) | 与 7B–27B 解码器持平 | LlamaGuard-3-8B |
| Response Safety | F1 | 与 7B–27B 解码器持平 | ShieldLM-14B |
| Jailbreak Detection (11策略) | F1 | competitive | WildGuard-7B |
| Throughput | 请求/秒 | **16×** 于 7B 基线 | LlamaGuard-3-8B |
| Latency | ms/请求 | **17×** 降低 | LlamaGuard-3-8B |
| 模型大小 | 参数量 | **0.3B** (23–90× 更小) | 7B–27B |

---

## 评分分解

| 维度 | 分数 | 满分 | 说明 |
|------|------|------|------|
| Innovation | 24 | 30 | Schema条件化多任务单pass设计，架构创新显著 |
| Experimental SOTA delta | 12 | 15 | 相当F1，23-90×更小，16×吞吐，17×延迟 |
| Experimental quality | 12 | 15 | 多任务多数据集全面评测，含吞吐/延迟对比 |
| Efficiency | 9 | 10 | 0.3B参数，16×吞吐，工业部署高度可行 |
| Generalization | 4 | 5 | 覆盖prompt/response/越狱/有害类别等多维度 |
| Domain relevance | 22 | 25 | 直接服务内容平台安全审核，高度相关 |
| **Total** | **83** | **100** | |
