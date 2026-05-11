## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **标题** | Valley3: Scaling Omni Foundation Models for E-commerce |
| **arXiv ID** | [2605.01278](https://arxiv.org/abs/2605.01278) |
| **提交日期** | 2026-05-02（v1），2026-05-06（v2） |
| **作者** | Zeyu Chen, Guanghao Zhou, Qixiang Yin, Ziwang Zhao, Huanjin Yao, Pengjiu Xia, Min Yang, Cen Chen, Minghui Qiu, et al. |
| **机构** | AIDC-AI（阿里巴巴国际数字商业集团 AI Center，Alibaba International Digital Commerce） |
| **领域** | E-commerce MLLM · Omni Multimodal · Short-Video · Audio-Visual · Agent |
| **Bucket** | STRONG ✦ |

---

## 方法概述 / Method Summary

Valley3 是专为全球化电商场景设计的全能多模态大语言模型（Omni MLLM），统一支持文本、图像、视频、音频四种模态的理解与推理。其核心贡献包括：

1. **四阶段电商持续预训练**：阶段一引入音频理解能力；阶段二训练跨模态指令遵循；阶段三注入电商领域知识（商品属性、商品描述、违规检测等）；阶段四实现长上下文推理。
2. **本地多语言音频能力**：专为短视频电商场景设计，支持在视频理解中同步处理语音（达人口播、商品讲解），对淘宝、速卖通等跨境电商直播场景有直接价值。
3. **可控推理模式**：支持一种非思考模式（快速响应）和三种深度思考模式（轻度/中度/重度 CoT），覆盖从实时审核到复杂商品研究的全场景。
4. **电商智能体搜索（Agentic Search）**：模型具备主动调用搜索工具的能力，可进行多步电商深度调研（如商品比价、竞品分析、合规检查）。
5. **全能电商基准（Omni E-commerce Benchmark）**：包含 6 项任务的内部综合测评，覆盖商品理解、多语言识别、违规内容检测、短视频摘要等。

### Story Arc
> **传统 VLM 被冻结的特征无法与推荐目标对齐** → Valley3 通过端到端四阶段预训练，将音频、电商领域知识与 Omni MLLM 深度融合，同时引入 Agentic Search 解锁复杂研究任务，实现了"全模态理解 + 深度推理 + 主动搜索"三位一体的电商基础模型。

---

## 创新性分析 / Innovation Analysis

**与先前工作的对比：**
- Valley2 （前代）：仅支持图文视频，缺乏原生音频，无 Agentic 能力
- 通用 MLLM（GPT-4o, Qwen2.5-VL 等）：缺乏电商专属预训练，短视频场景下口播识别与商品属性理解弱
- 工业级二阶段方法（预训练冻结特征 → 推荐模型）：特征对齐差，无法实时更新

**关键创新：**
1. 四阶段 Omni 预训练将音频与电商知识有机结合，是业界首个端到端 Omni 电商 MLLM
2. 可控思考链（轻/中/重 + 无思考）适配电商实时与离线场景的不同延迟需求
3. 在内部及开源电商基准上均超越强 baseline，同时在通用基准上保持竞争力

**可行性：** 高。AIDC-AI 有阿里国际电商海量数据支撑，四阶段预训练框架工程上已验证。

---

## 关键指标 / Key Metrics

| 数据集 / 任务 | 指标 | Valley3 | Best Baseline |
|-------------|------|---------|---------------|
| 内部电商 Omni Benchmark (6 tasks) | 综合得分 | SOTA | — |
| 开源电商 Benchmark | 综合得分 | SOTA | — |
| 通用多模态 Benchmark | 综合得分 | 竞争力 (competitive) | — |

> 具体数值未完全公开，论文中报告在全部 in-house 及开源电商基准上优于所有强 baseline。

---

## 评分 / Scoring

| 维度 | 得分 | 说明 |
|------|------|------|
| Innovation | 24/30 | 首个端到端 Omni 电商 MLLM，音频+短视频+Agent 三合一 |
| SOTA Delta | 12/15 | 在电商基准全面超越，但具体数值未充分公开 |
| Exp Quality / Ablations | 12/15 | 四阶段消融有据，开源+内部双重验证 |
| Efficiency | 7/10 | 可控推理模式降低延迟，但 Omni 模型本身较重 |
| Generalization | 4/5 | 通用基准保持竞争力 |
| Domain Relevance | 25/25 | 直接为电商达人内容生态构建，含违规检测、商品理解、短视频分析 |
| **总分** | **84/100** | ✦ 进入代码复现 |

---

## 代码复现 / Code Reproduction

见 `../code/Valley3/` — 完整实现包括：
- 简化 Omni MLLM 架构（Vision Encoder + Audio Encoder + MoE LLM）
- 四阶段训练脚本（含 toy 数据）
- 可控推理模式（非思考/轻度思考/重度思考）
- Agentic Search 工具调用 stub
- 电商 6-task 评估脚本
