# RAG-Enhanced LLMs for Dynamic Content Expiration Prediction in Web Search

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | RAG-Enhanced Large Language Models for Dynamic Content Expiration Prediction in Web Search |
| **Authors** | Tingyu Chen, Wenkai Zhang, Li Gao, Lixin Su, Ge Chen, Dawei Yin, Daiting Shi (Baidu Inc., Beijing) |
| **ArXiv** | https://arxiv.org/abs/2605.13052 |
| **Submitted** | 13 May 2026 |
| **Venue** | arXiv preprint; deployed at Baidu Search |
| **Code** | Not released |
| **Domain** | RAG, LLM for search, content freshness, content quality, industrial NLP |

---

## 方法概述 / Method Overview

### 故事弧线 / Story Arc

> **现有不足**: 商业搜索引擎中的内容新鲜度管理依赖静态时间窗口过滤，导致"一刀切"排名——内容可能时间上最新但语义上已过期，或语义上仍有效但被错误过滤。传统方法无法区分"查询意图相关的有效时间范围"。  
> **我们的设计**: 提出基于 LLM+RAG 的查询感知动态内容过期预测框架（Query-Aware Dynamic Content Expiration Prediction），将时效性重新定义为动态有效性推理任务——从文档中提取细粒度时间上下文，利用 LLM 推断每个查询的"语义有效期限"（validity horizon）。已在百度搜索线上部署。

### 技术细节 / Technical Details

**框架核心**:
1. **细粒度时间上下文提取**: 从文档中提取时间线索（发布时间、事件时间、更新标志等）
2. **查询感知有效期推断**: 利用 LLM 对查询的意图类型（高时效性 vs 低时效性）推断内容语义有效期
3. **RAG 增强**: 检索相关参考文档辅助 LLM 做时间推理，减少幻觉
4. **幻觉抑制机制**: 对比前向-后向 CoT（Contrastive Forward-Backward CoT）+ 来源权威性信号

**已部署线上A/B测试（百度搜索）**:
- 14天A/B测试
- High-Freshness 查询的 median day_away@4 下降 **12.81%**
- 搜索参与度正向提升

---

## 创新性分析 / Innovation Analysis

| 维度 | 分析 |
|------|------|
| **问题重新定义** | 将"内容新鲜度"从时间属性转变为"查询-语义有效期"，视角转换有创新性 |
| **RAG用于时间推理** | 利用 RAG 增强时间上下文，减少 LLM 时间推理幻觉，工程实用 |
| **工业部署验证** | 百度搜索线上A/B实验验证，结果可信度高 |
| **幻觉抑制** | 专为时间推理设计的 Contrastive CoT 机制，解决 LLM 对"过去/现在/未来"混淆问题 |
| **vs 先前工作** | 相比 Query-Freshness 静态方法，引入查询意图维度，更精细 |

---

## 关键指标 / Key Metrics

| 评测场景 | 指标 | 结果 |
|---------|------|------|
| 百度搜索 A/B (14天) | median day_away@4 (High-Freshness queries) | **-12.81%** |
| 搜索参与度 | 正向增益 | 正向 |

---

## 评分 / Scoring

| 维度 | 满分 | 得分 | 理由 |
|------|------|------|------|
| Innovation | 30 | 19 | 查询感知时效性重新定义有新意；Contrastive CoT 设计实用 |
| Experimental SOTA delta | 15 | 10 | 线上A/B实验结果可信，12.81% 提升对工业场景有意义 |
| Experimental quality/ablations | 15 | 9 | 工业场景验证充分，但离线实验细节略少 |
| Efficiency | 10 | 7 | 已在百度搜索部署，说明效率可接受 |
| Generalization | 5 | 3 | 仅百度搜索场景，跨平台泛化未验证 |
| **Domain relevance** | **25** | **17** | 内容有效期管理与电商内容质量治理相关；RAG for content understanding 直接可迁移 |
| **Total** | **100** | **65** | 工业落地价值高，方法论可迁移至电商内容质量评估 |
