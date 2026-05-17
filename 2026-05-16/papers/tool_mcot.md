# Tool-MCoT: Tool Augmented Multimodal Chain-of-Thought for Content Safety Moderation

## 基本信息 / Basic Info

| Field | Value |
|-------|-------|
| **Title** | Tool-MCoT: Tool Augmented Multimodal Chain-of-Thought for Content Safety Moderation |
| **Authors** | (Research team) |
| **Affiliation** | (Industry research lab) |
| **arXiv ID** | [2604.06205](https://arxiv.org/abs/2604.06205) |
| **Submission Date** | April 2026 |
| **Domain Tags** | `#content-moderation` `#VLM` `#chain-of-thought` `#tool-use` `#distillation` `#efficient-deployment` |
| **Total** | **72 / 100** |

---

## 故事弧线 / Story Arc

**现有问题:** 大型语言模型（LLM）在内容安全审核中效果出色，但其高计算成本和推理延迟使大规模部署面临严峻挑战——尤其是需要处理图文多模态输入的场景。

**设计方案:** 提出 **Tool-MCoT**，使用工具增强的多模态思维链数据训练小语言模型（SLM）。核心思路：LLM 教师生成"工具调用+思维链"蒸馏数据，SLM 学生在推理时**选择性调用外部工具**（OCR、图像描述、目标检测）以获取必要信息，从而在精度与效率之间取得平衡。

---

## 方法概述 / Method Overview

### 系统设计

```
Training Phase (Teacher → Student):
  LLM Teacher
  ├── Given: image + text input
  ├── Generates: tool call plan + CoT reasoning + final label
  └── Output: Tool-augmented CoT training data (SOFT LABELS)
       │
       ▼
  SLM Student
  ├── Fine-tuned on Tool-augmented CoT data
  ├── Learns: WHEN to call tools (selective invocation)
  └── Learns: HOW to reason from tool outputs

Inference Phase (Student only):
  Input: [Image + Text]
       │
       ▼
  SLM decides: need more info?
  ├── YES → Call tool(s): OCR / Captioning / Object Detection
  │           └── Incorporate tool output into reasoning
  └── NO  → Direct reasoning
       │
       ▼
  Moderation Decision
```

### 可用工具集
- **OCR**: 提取图像中的文字（品牌名、产品声明、违禁词）
- **Image Captioning**: 获取图像的自然语言描述
- **Object Detection**: 识别特定违规物体（违禁药品标志、武器等）

### 关键特性
- SLM **选择性**调用工具（不是每张图都调用），在精度与效率之间取得平衡
- 工具调用策略通过蒸馏学习，无需人工标注"何时调用工具"

---

## 关键指标 / Key Metrics

| Benchmark | Metric | Tool-MCoT (SLM) | Standard SLM | LLM Teacher |
|-----------|--------|-----------------|--------------|-------------|
| Open-source dataset 1 | F1 | Significant gain | Baseline | Upper bound |
| Open-source dataset 2 | F1 | Significant gain | Baseline | Upper bound |
| Open-source dataset 3 | F1 | Significant gain | Baseline | Upper bound |
| Inference efficiency | Latency | Much lower than LLM | Same as SLM | High |

---

## 创新性分析 / Innovation Analysis

- **工具增强 CoT 蒸馏**的组合在内容审核场景中具有新颖性
- 选择性工具调用（not always trigger）是实用工程创新，避免无意义开销
- 将 LLM 的推理过程（含工具使用规划）完整蒸馏给 SLM，不仅蒸馏结果

---

## 评分细项 / Scoring Breakdown

| Dimension | Score | Max | Justification |
|-----------|-------|-----|---------------|
| Innovation | 20 | 30 | 工具增强 CoT 蒸馏组合新颖；工具选择性调用实用 |
| Experimental SOTA Delta | 10 | 15 | 三数据集均有显著提升；具体数字受限 |
| Experimental Quality | 10 | 15 | 三数据集验证；工具消融合理 |
| Efficiency | 8 | 10 | SLM 大幅降低 LLM 推理成本 |
| Generalization | 3 | 5 | 三个开源数据集验证 |
| Domain Relevance | 21 | 25 | 内容安全审核；可扩展至电商场景 |
| **Total** | **72** | **100** | |
