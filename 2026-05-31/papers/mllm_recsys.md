# A General Framework for Multimodal LLM-Based Multimedia Understanding in Large-Scale Recommendation Systems

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **标题** | A General Framework for Multimodal LLM-Based Multimedia Understanding in Large-Scale Recommendation Systems |
| **arXiv ID** | 2605.09338 |
| **提交日期** | 2026-05-10 |
| **作者** | Yiming Zhu, Xu Liu, Ziyun Xu, Zheng Wu 等 |
| **机构** | Meta Platforms |
| **论文链接** | https://arxiv.org/abs/2605.09338 |
| **桶** | STRONG |
| **Total** | **69** |

---

## 方法概述 / Method

**故事弧（Story Arc）：**
> 多媒体内容（视频/图片/音频）蕴含丰富语义信号，但传统推荐系统无法充分利用这些高维信号，导致用户偏好建模失真。Multimodal LLM 能理解复杂多媒体内容，但直接部署在延迟敏感、超大规模推荐系统中面临计算挑战。本文提出一个面向工业大规模推荐的 **MLLM 多媒体理解通用框架**，解决延迟约束下的 MLLM 集成问题。

**核心设计（从搜索结果提炼）：**
1. **异步离线提取**：MLLM 离线对多媒体内容进行语义提取，结果缓存为结构化向量/描述
2. **轻量在线推理**：在线服务使用预提取的 MLLM 特征，保持低延迟
3. **统一语义空间**：不同模态（视频/图片/文本）的 MLLM 输出对齐到统一嵌入空间
4. **工业规模优化**：支持数亿级 item 库的多媒体语义索引

**与前工作差异：**
- 现有：端到端在线 MLLM（延迟高，不可扩展）
- 本文：离线 MLLM 提取 + 轻量在线推理（延迟可控，亿级规模）

---

## 关键指标 / Key Metrics

| 场景 | 指标 | 结果 |
|------|------|------|
| Meta 内部大规模推荐系统 | 参与度（engagement） | 有效提升 |
| 延迟约束 | 在线推理延迟 | 满足工业 SLA |

---

## 评分 / Scoring

| 维度 | 子分 | 说明 |
|------|------|------|
| Innovation (max 30) | 18 | 框架性工作，工程价值高；学术创新偏中等 |
| SOTA Δ (max 15) | 9 | Meta 内部系统验证，详细数字未完全公开 |
| Experimental Quality (max 15) | 10 | Meta 生产环境 A/B 测试 |
| Efficiency (max 10) | 8 | 离线 MLLM + 在线轻量推理的核心贡献 |
| Generalization (max 5) | 4 | 通用框架，适用多媒体内容推荐 |
| Domain Relevance (max 25) | 20 | 多媒体推荐，与电商短视频/直播高度相关 |
| **Total** | **69** | — |

---

## 创新性分析

1. **工业化路径**：解决了 MLLM 在大规模推荐系统中部署的核心工程挑战（延迟 vs 语义质量权衡）。
2. **离线-在线解耦**：将 MLLM 的计算成本摊销到离线阶段，在线只消费 MLLM 生成的紧凑特征——这是电商平台实际落地的可行路径。
3. **Meta 规模验证**：在全球最大规模社交/推荐平台之一验证，具有高度的工业参考价值。

---

## 电商 / 达人治理落地思路

- 电商短视频：MLLM 离线提取视频语义（商品、场景、情感）→ 在线推荐使用预提取特征
- 直播推荐：实时直播流按关键帧异步分析，快速更新用户-直播匹配特征
- 达人内容理解：批量分析达人视频内容，构建达人风格/品类嵌入库
