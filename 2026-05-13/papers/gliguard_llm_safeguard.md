# GLiGuard: Schema-Conditioned Classification for LLM Safeguard

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| 标题 | GLiGuard: Schema-Conditioned Classification for LLM Safeguard |
| 作者 | Fastino Labs 团队 |
| 机构 | Fastino Labs |
| arXiv | https://arxiv.org/abs/2605.07982 |
| GitHub | https://github.com/fastino-ai/GLiGuard |
| 索引确认日期 | **2026-05-13**（MarkTechPost 文章时间戳）|
| 领域标签 | LLM 安全护栏 · 内容审核 · AIGC 检测 · 双向编码器 · 内容治理 |
| 桶类别 | STRONG |
| 综合评分 | **74 / 100** |

---

## 方法概述 (中文)

现有 LLM 安全护栏（如 LlamaGuard、NemoGuard）通常基于**自回归解码器**，以"指令跟随生成"方式进行安全分类，存在两大缺陷：①延迟高（逐 token 生成）；②多维度安全属性需多次推理，无法并行。

**GLiGuard** 基于 GLiNER2 适配为**双向编码器（Bidirectional Encoder）**架构：
- **模型规模：** 0.3B 参数（约为 LlamaGuard4-12B 的 1/40）
- **Schema 条件化分类：** 输入指定的"模式化安全任务清单（Schema）"，模型在**单次前向传播**中同时完成：
  - Prompt 安全性判断
  - Response 安全性判断
  - 14类危害分类（Harm Category）
  - 11种越狱策略识别（Jailbreak Strategy）
- **硬决策组合：** 通过可配置的组合规则输出最终安全裁决
- **高效推理：** 双向注意力一次看完整输入，无需自回归，延迟降低 **16x**

---

## 故事线 (Story Arc)

> **现状不足：** 基于自回归解码器的护栏模型（LlamaGuard 系列）延迟高、成本大，无法满足高并发生产环境；多维度安全判断需多次调用，效率极低。
>
> **我们的解法：** GLiGuard 用双向编码器替换自回归解码器，Schema 条件化设计实现单次 forward pass 同时完成多个安全维度判断，在 23–90x 更小参数量下性能接近最强基线，延迟降低 16x。

---

## 创新点分析

| 维度 | 描述 |
|------|------|
| 核心创新 | 将 LLM 安全护栏从"生成式分类"转向"Schema 驱动的双向编码器多任务分类" |
| vs. 先前工作 | LlamaGuard/NemoGuard 用自回归生成做安全判断；本文单次 forward pass 处理所有维度 |
| 效率提升 | 16x 延迟下降，23–90x 参数量压缩，适合生产部署 |
| 开源 | 代码和模型已完全开源（GitHub: fastino-ai/GLiGuard）|
| 局限 | 专注文本安全，多模态扩展待探索；双向编码器在极长上下文的表现不如 decoder |

---

## 关键指标

| 基准 | 指标 | GLiGuard | 最强基线 |
|------|------|----------|---------|
| Prompt 安全基准（综合）| 平均 F1 | **87.7** | 最强 Prompt F1 基线 ≈ 89.4（差距 1.7）|
| Response 安全基准 | 平均 F1 | 第二名（所有开源护栏中）| LlamaGuard4-12B (第一) |
| vs. LlamaGuard4-12B | 参数量 | **0.3B** | 12B (40x 更大) |
| vs. NemoGuard-8B | 参数量 | **0.3B** | 8B (27x 更大) |
| 推理延迟 | 相对延迟 | **1x** | 解码器护栏 ≈ 16x |

---

## 评分分解

| 维度 | 得分 | 满分 | 说明 |
|------|------|------|------|
| 创新性 (Innovation) | 22 | 30 | 双向编码器 + Schema 条件化护栏设计新颖，工程价值高 |
| 实验 SOTA Delta | 11 | 15 | 87.7 avg F1，差距仅 1.7 F1，在 23–90x 小模型下成绩亮眼 |
| 实验质量/消融 | 11 | 15 | 对比 LlamaGuard4-12B、NemoGuard-8B 等多个强基线 |
| 效率 | 9 | 10 | 16x 延迟下降，参数量压缩 23–90x |
| 泛化性 | 4 | 5 | Schema 设计天然支持任务扩展 |
| 领域相关性 | 17 | 25 | LLM 安全护栏与电商内容治理关联，但非直接电商专用 |
| **总分** | **74** | **100** | — |

---

## 代码与数据

- GitHub：https://github.com/fastino-ai/GLiGuard（完全开源，代码+模型均可用）
- 模型：Hugging Face 上可下载
- 数据集：公开安全评测基准（SafeRLHF、HarmBench 等）
