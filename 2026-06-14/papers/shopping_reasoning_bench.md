## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **标题** | Shopping Reasoning Bench: An Expert-Authored Benchmark for Multi-Turn Conversational Shopping Assistants |
| **作者** | Shuxian Fan, Seonwoo Min, Youna Hu, Botao Xia, Jayakrishnan Unnikrishnan, Rowan Musselmann, Yifan Gao, Qingyu Yin, Priyanka Nigam, Bing Yin |
| **机构** | Amazon |
| **arXiv** | [2606.12608](https://arxiv.org/abs/2606.12608) |
| **数据集** | [HuggingFace: amazon/ShoppingReasoningBench](https://huggingface.co/datasets/amazon/ShoppingReasoningBench) |
| **发布日期** | 2026-06-12 |
| **领域标签** | 电商、对话式购物、多轮推理、评测/Benchmark、LLM-as-judge |

---

## 方法概述

**Story Arc**：对话购物助手已规模化上线，但现有评测要么是单轮、要么缺乏专家级 rubrics，导致模型看似"能答"但实际建议不够专业。本文由零售领域专家编写 **525 个 mission**（232 单轮 + 293 多轮），涵盖偏好澄清、跨商品权衡、兼容性判断等难以用唯一答案验证的能力，并为每个 mission 提供共计 **10,863 条带权重的二值 rubrics**，可用验证过的 LLM judge 做 criterion-level 细粒度评分。

Benchmark taxonomy 包含 5 大类 15 子类购物推理能力。Importance-weighted rubrics 区分 required（必须回答正确）与 above-and-beyond（专家额外能力）两档，使评测能区分"刚好合格"与"专家级"建议。

**Innovation vs Prior Work**：WebShop 等评测以单轮为主且聚焦搜索成功率，缺少复杂推理场景和专家标准。ShoppingReasoningBench 首次把"好建议"拆成可判定的细粒度原子 rubrics，并给出 judge 可靠性验证，为生产评测提供可落地范式。

---

## 关键指标

| 指标 | 值 | 说明 |
|------|-----|------|
| Overall weighted pass rate（9 模型） | 57.4%–77.2% | GPT-4o 系列居首 |
| Judge macro-F1 vs 专家 | **0.749**（κ=0.498） | judge 与人类专家高度一致 |
| 多轮退化（pass rate 下降） | **4–18 pp** | 随对话推进显著下降 |
| Optional vs Required rubrics gap | **13–29 pp** | 专业化程度有明确量化 |

---

## 评分

| 维度 | 得分 | 说明 |
|------|------|------|
| 方法创新性 | 22/30 | Rubric 体系 + judge 验证有新意，但技术层面是工程 benchmark 工作 |
| 实验指标 | 10/15 | 模型横向对比充分；无算法 SOTA 提升 |
| 实验质量 | 15/15 | 专家标注 + judge 验证 + 多轮分析 |
| 效率 | 6/10 | LLM judge 推理成本较高 |
| 泛化性 | 4/5 | 适用于任何对话购物助手评测 |
| 领域相关性 | 22/25 | 电商对话购物核心评测基础设施 |
| **Total** | **79/100** | 电商评测范式贡献显著，不满 80 因无新算法 |
