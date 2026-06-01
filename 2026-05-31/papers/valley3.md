# Valley3: Scaling Omni Foundation Models for E-commerce

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **标题** | Valley3: Scaling Omni Foundation Models for E-commerce |
| **arXiv ID** | 2605.01278 |
| **提交日期** | 2026-05-01 (模型权重发布日) |
| **作者** | Zeyu Chen, Guanghao Zhou, Qixiang Yin, Ziwang Zhao, Huanjin Yao, Pengjiu Xia, Min Yang, Cen Chen, Minghui Qiu 等 |
| **机构** | ByteDance Group |
| **论文链接** | https://arxiv.org/abs/2605.01278 |
| **官方代码** | https://github.com/bytedance/Valley |
| **本地复现** | code/Valley3/ |
| **桶** | STRONG |
| **Total** | **81** |

---

## 方法概述 / Method

**故事弧（Story Arc）：**
> 电商平台需要同时理解**文本 + 图片 + 视频 + 音频**（尤其是短视频/直播场景中的语音解说），现有 VLM 不支持音频，跨模态推理能力不足，且缺乏深度电商领域知识。Valley3 在 Qwen3-VL 骨干基础上，接入 Audio Transformer，通过**四阶段电商连续预训练 + 后训练**，打造全感官电商大模型，并配备可控推理（thinking/non-thinking 模式）与 Agentic Search 能力。

**架构：**
```
输入: 文本 / 图片 / 视频帧 / 音频
  ├─ Audio Transformer → 音频嵌入
  ├─ Vision Encoder (Qwen3-VL) → 视觉嵌入
  └─ Text Tokenizer → 文本 token
       ↓  MLP Connector (音频对齐)
  统一 LLM Backbone (Qwen3-VL 语言模型)
       ↓
  可控推理 (Non-thinking / Level-1/2/3 Thinking)
       ↓
  Agentic Search (按需调用搜索工具)
```

**四阶段训练：**
1. **Audio Understanding Stage** — 让模型从视觉-语言骨干学会音频语义
2. **Cross-Modal Instruction-Following Stage** — omni 多模态指令跟随
3. **E-commerce Domain Knowledge Stage** — 电商垂域知识注入
4. **Long-Context Reasoning Stage** — 长文本 + 深度推理

**与 Valley2 的差异：**
- 新增原生多语言音频能力（面向全球短视频电商）
- 可控推理级别（效率与深度权衡）
- Agentic Search（主动检索工具）

---

## 关键指标 / Key Metrics

| 评测集 | 任务数 | 结果 |
|--------|--------|------|
| 内部电商 benchmark (6 tasks) | 商品理解/视频描述/违规检测/搜索/问答/推荐 | 全面超越对比 baseline |
| 通用 benchmark | MLLM 标准评测 | 保持竞争力 |

> 注：完整数字表格在论文附录，本摘要来源于官方公告。

---

## 评分 / Scoring

| 维度 | 子分 | 说明 |
|------|------|------|
| Innovation (max 30) | 22 | Omni（含音频）+ 可控推理 + Agentic Search 组合新颖；骨干 Qwen3-VL 是已有工作 |
| SOTA Δ (max 15) | 12 | 6 任务全面超越 baseline，通用 benchmark 保持竞争力 |
| Experimental Quality (max 15) | 11 | 内部+开源双 benchmark，6 任务多样 |
| Efficiency (max 10) | 7 | 可控推理模式平衡简单场景效率 |
| Generalization (max 5) | 4 | 全球化（多语言音频），但强依赖电商场景 |
| Domain Relevance (max 25) | 25 | **电商原生** omni 大模型，覆盖全链路任务 |
| **Total** | **81** | — |

---

## 创新性分析

1. **原生音频 + 电商**：短视频电商中主播语音（价格、功效、促销词）是核心信息，首次将 audio understanding 深度融入电商 MLLM。
2. **Agentic Search**：支持模型在电商深度研究任务（如比价、商品溯源、规则查询）中主动调用检索工具，开启 AI Agent 新范式。
3. **可控推理**：三档 Thinking + Non-thinking 模式，简单场景（属性抽取）用低档次省推理成本，复杂场景（违规判定）用高档次保精度。

---

## 电商 / 达人治理落地思路

- **违规检测**：多模态 + 深度推理，能理解语音话术 + 商品图片 + 背景信息的联合违规
- **短视频理解**：图文音频三路对齐，支持商品描述生成、卖点提取、内容质量评估
- **达人内容审核**：Agentic Search 可主动查询平台规则库，进行规则驱动的内容合规判断
