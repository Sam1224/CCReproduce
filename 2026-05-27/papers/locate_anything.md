# LocateAnything: Fast and High-Quality Vision-Language Grounding with Parallel Box Decoding

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | LocateAnything: Fast and High-Quality Vision-Language Grounding with Parallel Box Decoding |
| **Authors** | Bharath Raj Nagoor Kani, Noah Snavely et al. |
| **Affiliations** | NVIDIA |
| **arXiv** | [2605.27365](https://arxiv.org/abs/2605.27365) |
| **Submitted** | 2026-05-26 (listed 2026-05-27 GMT+8) ✓ |
| **Bucket** | WEAK |
| **Total** | **69 / 100** |

---

## 方法概述 / Method Overview

### EN
Vision-language grounding models (e.g., Grounding DINO) typically encode bounding boxes as sequences of coordinate tokens decoded one-by-one, which (1) creates a geometry mismatch by treating coupled x,y,w,h independently, and (2) bottlenecks inference through sequential generation. **LocateAnything** introduces **Parallel Box Decoding (PBD)**: all bounding box coordinates are decoded simultaneously as an *atomic unit* in a single forward step. The model adopts a generalist design supporting referring expression grounding, multi-object detection, GUI element grounding, and text localization. NVIDIA built a **large-scale data engine** curating **LocateAnything-Data** with 138M+ training samples across diverse domains. The result is both higher throughput and improved localization accuracy, particularly in complex and cluttered scenes.

### ZH
视觉语言定位模型通常将边界框编码为坐标 Token 序列逐一解码，(1) 将耦合的 x,y,w,h 独立处理，破坏几何一致性；(2) 串行生成造成推理瓶颈。**LocateAnything** 提出**并行边界框解码（PBD）**：所有坐标在单次前向传播中同步解码为**原子单元**。模型采用通才设计，支持指代表达定位、多目标检测、GUI 元素定位、文字定位等任务。NVIDIA 构建了包含 **1.38 亿+训练样本**的大规模数据集 LocateAnything-Data。结果是吞吐量更高、定位精度更好，尤其在复杂杂乱场景下。

---

## 故事弧 / Story Arc

> **"串行坐标 Token 解码慢且几何不一致"** → PBD 并行原子解码边界框，一次前向完成，精度与速度双提升，1.38 亿样本数据引擎支撑通才多任务定位模型。

---

## 创新性分析 / Innovation Analysis

1. **并行解码**：将 bbox 坐标作为原子单元并行输出，概念简洁、效果显著。
2. **几何一致性**：x,y,w,h 作为整体优化，避免坐标间独立误差累积。
3. **大规模数据引擎**：1.38 亿样本，工业级 VLM 训练基础设施。
4. **通才设计**：单一模型覆盖多种定位任务，具有强泛化性。

---

## 关键指标 / Key Metrics

| Task | Metric | LocateAnything | Baseline |
|------|--------|---------------|---------|
| Referring Expression Grounding | Acc | SOTA | — |
| GUI Grounding | Acc | SOTA | — |
| Text Localization | F1 | SOTA | — |
| Inference throughput | bbox/s | Significantly faster | Seq. decoding |

---

## 评分明细 / Scoring Breakdown

| 维度 | 分值 | 得分 | 说明 |
|------|------|------|------|
| Innovation | 30 | 22 | PBD 概念简洁新颖，几何原子解码思路优雅 |
| Experimental SOTA delta | 15 | 11 | 多任务 SOTA，速度提升明显 |
| Experimental quality / ablations | 15 | 12 | 大规模数据，多任务评测 |
| Efficiency | 10 | 9 | 并行解码大幅提升吞吐量 |
| Generalization | 5 | 5 | 多任务通才模型 |
| Domain relevance | 25 | 10 | VLM 定位对电商图文理解有间接价值 |
| **Total** | **100** | **69** | |

---

## 电商治理价值 / E-commerce Governance Value

- **商品细粒度定位**：可用于识别商品图片中的品牌 LOGO、违禁成分标注、价格牌
- **直播截图理解**：定位直播画面中的商品区域、弹幕违规词等
- **GUI 操作自动化**：Agent 与电商 App 交互时精确点击商品元素
