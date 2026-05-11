## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **标题** | From Static Analysis to Audience Dissemination: A Training-Free Multimodal Controversy Detection Multi-Agent Framework |
| **arXiv ID** | [2605.02939](https://arxiv.org/abs/2605.02939) |
| **提交日期** | 2026-05（May 2026） |
| **作者** | Zihan Ding, Ziyuan Yang, Yi Zhang |
| **机构** | (未完全公开，来自中国高校/研究院) |
| **领域** | Content Governance · Multimodal Controversy Detection · Multi-Agent · Social Video Platforms |
| **Bucket** | STRONG |

---

## 方法概述 / Method Summary

AuDisAgent 将多模态争议内容检测（MCD）从静态特征学习重新定义为"内容传播动力学"问题。框架核心：

1. **三专业筛查 Agent**：
   - Video Agent：从视觉角度初步评估视频内容争议性
   - Comment Agent：从文本角度分析评论情感与敌意
   - Interaction Agent：从跨模态交互角度捕获视频与评论的一致/矛盾关系
2. **Viewing Panel Agent**：当三 Agent 无法达成共识时激活，模拟具有多元背景和立场的观众群体进行讨论，输出最终判定。
3. **无需训练（Training-Free）**：全程基于 GLM4-9B（支持 128K token 长文本推理）零样本推理，无需微调。
4. **评测数据集**：公开 MMCD 数据集（10,000+ 中文视频，含丰富社交评论上下文）。

### Story Arc
> **MCD 作为静态特征学习任务无法捕捉不同受众群体的多元视角** → AuDisAgent 将内容传播过程建模为多 Agent 动态协作，引入"受众观看小组"模拟真实用户评审流程，在富评论和少评论两种场景下均超越 13 个强 baseline。

---

## 创新性分析 / Innovation Analysis

**与先前工作对比：**
- 传统 MCD 方法：从视频+评论提取联合特征 → 二分类，忽视受众多样性
- AgentMCD（最强 baseline）：多 Agent 但缺乏受众模拟与动态协商机制
- AuDisAgent：引入"受众传播"范式，Viewing Panel 模拟不同背景观众的讨论过程

**关键创新：**
1. 将 MCD 从监督分类转为基于传播动力学的多 Agent 推理
2. Viewing Panel Agent 的受众背景多样化模拟是核心创新点
3. 完全训练无关，可直接迁移至新平台/新类型内容

**可行性：** 高。基于成熟 LLM，框架轻量，无需标注数据。

---

## 关键指标 / Key Metrics

| 数据集 | 场景 | 指标 | AuDisAgent | 第二名 |
|--------|------|------|-----------|--------|
| MMCD | 富评论 | Accuracy | 领先 +1.33% | AgentMCD |
| MMCD | 富评论 | F1-Score | 领先 +1.68% | AgentMCD |
| MMCD | 少评论 | Accuracy | 领先 +1.10% | AgentMCD |
| MMCD | 少评论 | F1-Score | 领先 +2.00% | AgentMCD |

---

## 评分 / Scoring

| 维度 | 得分 | 说明 |
|------|------|------|
| Innovation | 20/30 | 受众传播范式创新，但 LLM 多 Agent 框架不罕见 |
| SOTA Delta | 10/15 | 小幅但稳定的性能提升，13 个基准全面超越 |
| Exp Quality / Ablations | 10/15 | 双场景评测，但仅一个数据集 |
| Efficiency | 8/10 | 无需训练，单模型即可部署 |
| Generalization | 3/5 | 仅在 MMCD 上验证，中文视频场景 |
| Domain Relevance | 21/25 | 社交视频平台内容风险管理，直接适用电商短视频审核 |
| **总分** | **72/100** | Feishu 推送 |
