# GLiNER Guard: Unified Encoder Family for Production LLM Safety and Privacy

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | GLiNER Guard: Unified Encoder Family for Production LLM Safety and Privacy |
| **Authors** | HiveTraceLab team |
| **ArXiv** | https://arxiv.org/abs/2605.05277 |
| **Submitted** | 6 May 2026 |
| **Venue** | arXiv preprint |
| **Code** | GitHub (HiveTraceLab) |
| **Domain** | LLM safety, PII detection, content moderation, efficient inference |

---

## 方法概述 / Method Overview

### 故事弧线 / Story Arc

> **现有不足**: 生产环境中的 LLM 安全流水线通常需要多个独立模型分别处理安全分类和 PII 检测，运维成本高、延迟叠加。  
> **我们的设计**: GLiNER Guard 是一个统一编码器家族，在单次前向传播中同时完成安全分类和 PII span 级别检测，三个变体（145M compact uni/bi-encoder + 209M Omni）覆盖不同吞吐量需求。

### 技术细节 / Technical Details

**架构**:
- 共享文本编码器 + span-scoring heads（提取任务）+ classification heads（分类任务）
- 基于 GLiNER2 构建，统一 span-level 隐私提取和 label-level 安全分类
- 三个变体:
  - **GLiNER Guard compact uni-encoder** (145M): 最高吞吐量
  - **GLiNER Guard compact bi-encoder** (147M): 检索优化
  - **GLiNER Guard Omni** (209M): 最强审核质量

**训练数据**:
- 467,273 条多任务训练样本
- 5 类分类任务: 安全分类、对抗攻击检测、有害内容分类、意图识别、语气分类
- 1 类提取任务: PII span 检测

**新资源**:
- PII-Bench: span 级别 PII 检测端到端评测基准

---

## 创新性分析 / Innovation Analysis

| 维度 | 分析 |
|------|------|
| **任务统一** | 安全分类 + PII 检测在单模型中实现，减少 1/3 以上的推理调用 |
| **多变体设计** | 145M/209M 覆盖不同延迟-质量权衡需求 |
| **PII-Bench** | 首个 span 级别 PII 端到端评测基准，填补领域空白 |
| **生产就绪** | 193 req/s @ A100，P99 < 1s，直接可部署 |
| **vs GLiGuard** | 与 GLiGuard 互补——GLiNER Guard 额外覆盖 PII 检测 |

---

## 关键指标 / Key Metrics

| 评测 | 指标 | GLiNER Guard Omni | 竞品 |
|------|------|-------------------|------|
| Safety benchmarks | Accuracy | Competitive w/ 7B+ | GLiGuard, LlamaGuard |
| PII-Bench | F1 | SOTA on span-level | Previous NER models |
| Throughput (compact) | req/s | **193** | ~12 (7B models) |

---

## 评分 / Scoring

| 维度 | 满分 | 得分 | 理由 |
|------|------|------|------|
| Innovation | 30 | 18 | 任务统一设计合理；PII-Bench 贡献有价值 |
| Experimental SOTA delta | 15 | 9 | 精度相当，效率提升明显 |
| Experimental quality/ablations | 15 | 9 | 多任务评测，三变体对比 |
| Efficiency | 10 | 9 | 参数效率是核心亮点 |
| Generalization | 5 | 3 | 英文为主 |
| **Domain relevance** | **25** | **12** | 通用 LLM 安全，与电商内容治理相关但非电商专用 |
| **Total** | **100** | **60** | 工程价值明确，适合平台级 LLM 安全基础设施 |
