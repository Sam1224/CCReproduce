## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | OneReason Technical Report |
| **Authors** | OneRec Team (83+ authors from Kuaishou Inc. & ByteDance) |
| **Affiliations** | Kuaishou Inc., ByteDance |
| **ArXiv ID** | [2606.06260](https://arxiv.org/abs/2606.06260) |
| **Submitted** | 2026-06-04 (indexed 2026-06-07 GMT+8) |
| **Categories** | cs.IR, cs.AI, cs.LG |
| **Code** | No official code (technical report) |
| **Bucket** | STRONG |
| **Total** | **79 / 100** |

---

## Score Breakdown

| Dimension | Score | Max | Justification |
|-----------|-------|-----|---------------|
| Innovation | 27 | 30 | Novel finding that naive thinking-mode fails in generative rec; three-level cognition-enhanced CoT + Fast-Slow online architecture are concrete, deployable innovations |
| Experimental SOTA delta | 13 | 15 | Online A/B: Conv +12.643%, Click +1.709%, CVR +1.865%, CTR +0.519%; low-active user revenue +13.323% |
| Experimental quality / ablations | 11 | 15 | Large-scale industrial A/B; multiple Kuaishou service lines; user-stratified analysis |
| Efficiency | 7 | 10 | Fast-Slow architecture explicitly addresses latency; retrieval share (27.2%) shows selective reasoning |
| Generalization | 4 | 5 | Deployed across short-video, live-streaming, ads, e-commerce at Kuaishou |
| Domain relevance | 17 | 25 | Core e-commerce/content recommendation technology; directly deployed in production |
| **Total** | **79** | **100** | |

---

## 方法概述 / Method Overview

### 问题背景（故事弧）
快手（Kuaishou）与字节跳动（ByteDance）部署的生成式推荐系统（OneRec 家族）已广泛应用于短视频、直播、广告和电商场景，并能从规模扩展（scaling）中受益。然而，现有生成式推荐模型的推理能力难以激活：由于推荐的基本单元是"item token"（不带语义的 ID），无法像 LLM 一样构造有意义的 Chain-of-Thought（CoT）序列，导致"先思考再推荐"的范式不能直接迁移。

**X is insufficient → we design Y to solve it：**
> 直接在生成式推荐中引入 thinking-mode（仿照 LLM 的 reasoning CoT）并不奏效：preliminary 研究（OneRec-Think、OpenOneRec）发现思考模式相比非思考模式没有优势，甚至更差。核心原因是 item token 缺乏语言语义感知。OneReason 提出：先在预训练阶段强化 item token 的语义感知（perception），再构建三层认知增强 CoT 格式（cognition-enhanced CoT），最终用 specialize-then-unify 的 RL 训练配方让推荐推理真正有效，并辅以 Fast-Slow Thinking 在线架构平衡效果与延迟。

### 核心方法

1. **感知增强预训练（Perception-enhanced Pre-training）**：将 item token 与其对应的自然语言描述进行对齐，使 item token 具备足够的语义感知能力，为后续 CoT 推理提供基础。

2. **三层认知增强 CoT（Three-level Cognition-enhanced CoT）**：
   - 第一层：用户意图识别（识别当前 session 的核心兴趣）
   - 第二层：候选 item 的多维度评估（相关性、新颖性、多样性）
   - 第三层：最终推荐决策与理由生成

3. **Specialize-then-Unify RL 训练配方**：先在各业务场景分别 RL 微调（specialize），再统一合并（unify），解决多业务推理目标不一致问题。

4. **Fast-Slow Thinking 在线架构**：慢思考（slow thinking）路径处理高价值、需要推理的请求；快路径（fast）处理对延迟敏感的请求，Retrieval Share 约 27.2%。

### English Summary

OneReason is a technical report from the OneRec team at Kuaishou/ByteDance that investigates how to enable genuine reasoning in generative recommendation models. The key finding is that naive application of the LLM "think-before-answer" paradigm fails because item tokens lack linguistic semantics, making meaningful CoT construction impossible. OneReason addresses this through a three-stage solution: (1) perception-enhanced pre-training to give item tokens semantic grounding; (2) a three-level cognition-enhanced CoT format covering user intent → item evaluation → decision; (3) a specialize-then-unify RL recipe for multi-service training. A Fast-Slow Thinking online architecture balances reasoning quality with latency constraints. Online A/B results show significant gains across multiple Kuaishou service lines.

---

## 创新点分析 / Innovation Analysis

**中文：** 最重要的发现是"为什么 thinking-mode 在生成式推荐中失效"——item token 的语义空洞是根本原因，而不是 CoT 格式问题或 RL 训练问题。基于此诊断，感知增强预训练 + 三层 CoT 格式 + specialize-then-unify RL 的组合方案是有理论依据的系统性解决方案。Fast-Slow 双路在线架构也是对工业部署中"效果-延迟 Pareto 前沿"的直接回应，具有较高的工程复现价值。

**English:** The most important contribution is the diagnostic: item token semantic poverty is the root cause of thinking-mode failure in generative recommendation, not the CoT format or RL training. The perception-enhanced pre-training → three-level CoT → specialize-then-unify RL pipeline follows from this diagnosis. The Fast-Slow online architecture is a practically grounded response to the quality-latency tradeoff in large-scale production recommendation systems.

**Limitations:** As a technical report, reproducibility outside Kuaishou's infrastructure is limited. The exact CoT format and RL reward design details may not be fully disclosed.

---

## 关键指标 / Key Metrics

| Setting | Metric | Value | Notes |
|---------|--------|-------|-------|
| Online A/B, Combined | Conv | **+12.643%** | Multi-turn conversion |
| Online A/B, Combined | Click | **+1.709%** | |
| Online A/B, Combined | CVR | **+1.865%** | Conversion rate |
| Online A/B, Combined | CTR | **+0.519%** | |
| Online A/B, Combined | Retrieval Share | **27.2%** | % requests using slow path |
| User stratified, Low-Active | Revenue | **+13.323%** | Largest gain for low-activity users |
