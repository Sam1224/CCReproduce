# E-VAds: An E-commerce Short Videos Understanding Benchmark for MLLMs

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **arXiv** | https://arxiv.org/abs/2602.08355 |
| **提交日期** | 2026-02（catch-up 发现；不在 2026-06-04 主窗口，但领域高度契合，纳入补充） |
| **作者** | Xianjie Liu, Yiman Hu, Liang Wu, Ping Hu, Yixiong Zou, Jian Xu, Bo Zheng |
| **机构** | Alibaba Group (Taobao) |
| **领域标签** | `E-commerce Video` `MLLM Benchmark` `Product Understanding` `Multi-agent Annotation` `电商短视频` |
| **桶位** | STRONG |

---

## 方法概述

E-commerce 短视频（商品推广视频、广告视频）是电商平台高增长、高变现的内容形态，但**目前所有 MLLM 评测基准均面向通用视频**，忽视了电商视频特有的目标驱动（goal-driven）格式与超密集多模态信号（高密度文字叠加、快速切镜、商品特写 + 达人解说 + 背景音乐）。

E-VAds 提出了一套**多模态信息密度评估框架（Multimodal Information Density Assessment Framework）**，量化了电商视频在视觉、音频、文本三个模态上的复杂度，证明其显著高于 mainstream 数据集（ActivityNet、MSVD 等）。在此分析基础上，构建了 E-VAds（E-commerce Video Ads Benchmark）：

- **数据规模**：从淘宝抓取 **3,961 个高质量视频**，覆盖多个商品类目（美妆、服饰、数码、食品、家居等）
- **问题规模**：利用 **multi-agent 系统**自动生成 **19,785 个开放式问题**，覆盖商业意图理解、商品属性推理、场景情感、行为召唤（CTA）识别等维度
- **评测对象**：系统评估 GPT-4o、Gemini 1.5 Pro、Claude、LLaVA-Next、Qwen-VL 等主流 MLLM

---

## 故事弧线 / Story Arc

**MLLM 评测基准聚焦通用视频，忽视电商视频的高密度多模态结构** → E-VAds 用信息密度框架量化这一 gap，发现电商视频在所有模态均显著更复杂 → 构建首个专为电商视频设计的基准 → 揭示当前主流 MLLM 在商业意图理解上存在显著不足，为后续模型改进提供方向。

---

## 创新性分析

- **首个电商视频 MLLM 基准**：填补了 benchmark 领域的明显空白，社区价值清晰。
- **多模态信息密度框架**：提出量化三模态（视觉 / 音频 / 文本）信息密度的方法，为后续 adaptive benchmark 设计提供基础工具。
- **Multi-agent 问题生成**：利用 LLM/MLLM 多 agent 协作自动生成高质量开放式问题，兼顾规模与质量，可借鉴到其他领域基准构建。
- **商业意图维度**：CTA 识别、产品卖点提取、达人推介效果评估等问题类型在通用 VQA 基准中基本缺失。
- **局限**：数据全部来自淘宝，平台多样性有限；问题由 LLM 自动生成，可能存在系统性偏差。

---

## 关键指标

| 数据集 | 评测维度 | 最佳 MLLM | 人类参考 |
|--------|---------|-----------|---------|
| E-VAds (3,961 视频, 19,785 Q) | 商业意图理解 | GPT-4o (最优，具体数据论文中) | 显著高于 MLLM |
| E-VAds | 跨模态密度评分 | — | 所有 MLLM 均在高密度场景下退化 |

- 论文摘要未公布具体百分点；以上结论基于 search snippet；完整数字需查阅原文。

---

## 评分 (满分 100)

| 维度 | 分 / 满分 | 理由 |
|------|----------|------|
| 创新性 | 22 / 30 | 首个电商视频 MLLM 基准 + 信息密度框架，benchmark 工作创新性上限受限 |
| 实验指标 | 11 / 15 | 覆盖主流 MLLM；无单一 SOTA delta 数字 |
| 实验质量 | 12 / 15 | 数据规模大、类目广、多 agent 标注流程严谨 |
| 效率 | 6 / 10 | 多 agent 标注流程需较大 LLM 调用开销 |
| 泛化性 | 5 / 5 | 评测多个主流 MLLM，结论通用 |
| 相关性 | 22 / 25 | 淘宝/抖音/快手电商短视频理解，与达人内容质量评估直接相关 |
| **Total** | **78** |

> **日期注**: 本文提交于 2026-02，非 2026-06-04 主窗口，为 catch-up 发现。
