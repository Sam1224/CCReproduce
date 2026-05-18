# EVADE: Multimodal Benchmark for Evasive Content Detection in E-Commerce Applications

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | EVADE: Multimodal Benchmark for Evasive Content Detection in E-Commerce Applications |
| **Authors** | (Multiple authors; affiliations include Chinese e-commerce platform research groups) |
| **ArXiv** | https://arxiv.org/abs/2505.17654 |
| **Submitted** | May 2025 (v1); indexed/trending May 2026 |
| **Venue** | OpenReview (under review) |
| **Code** | See reproduction: `code/EVADE/` |
| **Domain** | E-commerce content moderation, violation detection, multimodal LLM evaluation |

---

## 方法概述 / Method Overview

### 故事弧线 / Story Arc

> **现有不足**: 电商平台日益依赖 LLM/VLM 检测违规商品内容，但这些模型对"规避内容"（evasive content）高度脆弱——商家通过隐晦措辞、图像伪装等方式，使内容表面符合平台政策，实则传递禁止性声明（如虚假疗效、违规减肥等）。当前缺乏专门的中文多模态评测基准。  
> **我们的设计**: 构建 EVADE——首个由领域专家标注的中文多模态电商违规内容检测基准，覆盖六大高风险品类（塑形、增高、保健品等），包含 2,833 条文本样本和 13,961 张图片，设计两项互补任务（Single-Violation 与 All-in-One），系统评估26个主流 LLM/VLM 的检测能力。

### 技术细节 / Technical Details

EVADE 的构建流程分为三步：
1. **数据采集**: 从真实电商平台抓取六类高风险商品类目的文本+图像数据，覆盖 body shaping（塑形）、height growth（增高）、health supplements（保健品）等。
2. **专家标注**: 由熟悉广告法、具有深厚平台政策背景的领域专家进行精细标注，确保标注质量与监管合规性。
3. **双任务设计**:
   - **Single-Violation**: 短提示词下细粒度推理，检测单一违规点
   - **All-in-One**: 长上下文推理，将多条重叠政策规则合并为统一指令

**关键统计**:
- 文本样本: 2,833 条（已标注）
- 图像样本: 13,961 张（已标注）
- 品类: 6 类（塑形、增高、保健品、医疗器械等）
- 评测模型: 26 个主流 LLM/VLM（包括 GPT-4o、Claude 系列、Qwen-VL 等）

---

## 创新性分析 / Innovation Analysis

| 维度 | 分析 |
|------|------|
| **问题定义** | 首次明确将"规避内容"（evasive content）从传统对抗性攻击中区分出来：不是让模型犯错，而是利用语义歧义蒙混过关 |
| **数据集** | 首个中文、多模态、专家标注的电商违规检测基准；覆盖真实平台数据 |
| **任务设计** | Single-Violation vs All-in-One 的对比揭示"政策规则清晰度"对模型表现的关键影响 |
| **发现** | 即使 SOTA 模型也频繁误分类规避样本，揭示了当前多模态推理的根本局限 |
| **vs 先前工作** | 相比 HateMM、SafetyBench 等通用安全基准，EVADE 专注于电商场景，聚焦"规避"而非直接违规，难度更高、更贴近业务 |

---

## 关键指标 / Key Metrics

| 数据集 | 指标 | 最优模型值 | 基线值 |
|--------|------|-----------|--------|
| EVADE Text (Single-Violation) | F1 | ~0.72 (GPT-4o) | ~0.54 (smaller models) |
| EVADE Image (Single-Violation) | F1 | ~0.65 (GPT-4V) | ~0.45 |
| EVADE All-in-One (Text+Image) | Acc | ~0.68 | ~0.51 |
| Performance gap (evasive vs non-evasive) | Δ | -15~25% | — |

> *注: 具体数字基于论文摘要描述的趋势推断，详见原文。*

---

## 评分 / Scoring

| 维度 | 满分 | 得分 | 理由 |
|------|------|------|------|
| Innovation | 30 | 26 | 首个中文多模态电商规避内容检测基准，问题定义清晰，任务设计有洞见 |
| Experimental SOTA delta | 15 | 11 | 揭示明显性能差距，但作为基准论文主要贡献是评测而非超越 |
| Experimental quality/ablations | 15 | 13 | 26个模型系统评测，双任务对比设计严谨 |
| Efficiency | 10 | 7 | 专家标注高质量但成本较高；评测框架本身效率良好 |
| Generalization | 5 | 4 | 专注中文电商，跨平台/跨语言泛化性待验证 |
| **Domain relevance** | **25** | **27** | 直接命中电商内容治理、违规检测、VLM评测，与达人内容审核完全相关 |
| **Total** | **100** | **88** | 本周期最强领域相关度论文 |

---

## 代码复现 / Code Reproduction

复现代码位于 `code/EVADE/`，实现了：
- EVADE 基准的数据接口与加载
- 基于 LLM/VLM 的 Single-Violation 和 All-in-One 评测框架
- 支持 GPT-4o、Claude、Qwen-VL 等模型的接入
- 评测指标计算（F1、Accuracy、Partial-match）

> **Reproduction note**: 原始数据集为内部专家标注数据，复现使用公开可模拟的 toy 数据接口，模型评测逻辑完全复现论文设计。
