# Towards On-Policy Data Evolution for Visual-Native Multimodal Deep Search Agents (ODE)

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | Towards On-Policy Data Evolution for Visual-Native Multimodal Deep Search Agents |
| **arXiv ID** | 2605.10832 |
| **Submitted** | 2026-05-08 (indexed within May 2026 window) |
| **Link** | https://arxiv.org/abs/2605.10832 |
| **Authors** | (details from paper) |
| **Affiliation** | TBC |
| **Code** | Not yet public |
| **Venue** | arXiv preprint |
| **Bucket** | WEAK |

---

## 方法概述 / Method Overview

**问题（Story Arc）：** 多模态深度搜索 Agent 在使用工具（搜索、OCR、图像增强）检索视觉证据时，中间返回的图像被丢弃而无法被后续工具复用；同时，训练数据通常由静态策略生成，无法随着模型能力演进而更新，导致数据与策略能力之间持续存在"分布漂移"。

**解决方案（两大创新）：**  
1. **Image Bank Reference Protocol（图像库引用协议）**：将每个工具返回的图像注册为全局可寻址引用（addressable reference），使中间视觉证据可被后续工具复用，实现真正的"视觉原生"（visual-native）多跳推理。
2. **On-Policy Data Evolution（ODE，在线策略数据演化）**：通过闭环生成器，在每轮训练后用当前策略的 rollout 反馈（成功/失败轨迹）自动精炼训练数据，而非依赖静态合成数据集。

---

## 创新性分析 / Innovation Analysis

| 维度 | 分析 |
|------|------|
| vs. OpenSearch-VL / MTA-Agent | Image bank 是新颖的视觉中间状态管理机制，超越现有工具调用框架 |
| vs. 静态 SFT 数据 | ODE 闭环演化使训练数据始终贴合当前策略能力 |
| 实验支撑 | 8 个多模态深度搜索 benchmark 验证 |

---

## 关键指标 / Key Metrics

| Benchmark Suite | Metric | ODE | Static SFT |
|-----------------|--------|-----|-----------|
| 8 Multimodal Search Benchmarks (avg) | Accuracy | +X% (paper specific) | Best static |
| Complex multi-hop tasks | Accuracy | substantial improvement | Image-discarding agents |

---

## 评分明细 / Scoring Breakdown

| 维度 | 满分 | 得分 | 说明 |
|------|------|------|------|
| Innovation | 30 | 20 | 图像库引用协议新颖；ODE 是对 self-play 的实用工程化 |
| SOTA Delta | 15 | 10 | 8 benchmark 改善，但具体数值待核实 |
| Experimental Quality | 15 | 10 | 多 benchmark + ablation |
| Efficiency | 10 | 5 | ODE 闭环有训练成本 |
| Generalization | 5 | 3 | 多搜索场景 |
| Domain Relevance | 25 | 14 | 搜索 agent 可迁移电商商品搜索/内容溯源 |
| **Total** | **100** | **62** | WEAK |

---

## 故事弧 / Story Arc

> "多模态搜索 agent 丢弃中间视觉证据且训练数据与能力失配 → Image Bank 协议使中间图像可复用，ODE 闭环演化保持数据与策略同步，在 8 个 benchmark 上超越静态训练 agent。"
