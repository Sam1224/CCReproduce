## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | Revising Context, Shifting Simulated Stance: Auditing LLM-Based Stance Simulation in Online Discussions |
| **Authors** | Xinnong Zhang, Wanting Shan, Hanjia Lyu, Zhongyu Wei, Jiebo Luo |
| **Affiliations** | Fudan University, University of Rochester |
| **ArXiv ID** | [2606.06443](https://arxiv.org/abs/2606.06443) |
| **Submitted** | 2026-06-04 (indexed 2026-06-07 GMT+8) |
| **Categories** | cs.CL, cs.SI |
| **Bucket** | WEAK |
| **Total** | **61 / 100** |

---

## Score Breakdown

| Dimension | Score | Max | Justification |
|-----------|-------|-----|---------------|
| Innovation | 18 | 30 | Counterfactual context revision as audit framework + meme-based multimodal context extension are interesting but incremental |
| Experimental SOTA delta | 8 | 15 | Baseline accuracy 77.64%; demonstrates context sensitivity but modest absolute gains |
| Experimental quality / ablations | 10 | 15 | 1,821 Reddit conversations; multiple revision strategies; text-only vs. multimodal comparison |
| Efficiency | 7 | 10 | Lightweight audit methodology; no model training required |
| Generalization | 3 | 5 | Reddit-specific; revision strategies transferable |
| Domain relevance | 15 | 25 | Useful for understanding LLM bias in stance simulation; indirect relevance to influencer content audit and opinion analysis |
| **Total** | **61** | **100** | |

---

## 方法概述 / Method Overview

### 问题背景（故事弧）
LLM 被越来越多地用于"立场模拟"（Stance Simulation）——模拟特定用户在特定话题上的观点，用于社会科学研究和意见分析。然而，LLM 模拟的立场是否真正反映了该用户的特定信念，还是对对话上下文的变化高度敏感（即"上下文偏见"），缺乏系统性审计。

**X is insufficient → we design Y to solve it：**
> 现有对 LLM 立场模拟的评测只关注准确率，不关注模拟结果对语义无关的上下文变化是否稳定。本文提出"反事实上下文修订"审计框架：在原对话上下文中模拟用户立场，然后通过受控的文本修订（或引入 meme 的多模态修订）改变上下文，再次模拟立场，测量立场转移率和方向性偏移，量化 LLM 立场模拟的上下文敏感性。

### 核心方法

1. **反事实上下文修订**：保持讨论话题不变，通过多种受控策略（强调、去强调、情绪化、去情绪化）修订对话上下文文本。

2. **多模态上下文（Meme）**：在文本修订基础上引入 meme 图片，测量视觉上下文相对于文本描述的额外影响。

3. **实验设置**：1,821 段 Reddit 讨论（涉及 DeepSeek/Claude/Llama），原始立场模拟准确率 77.64（Acc）/ 78.10（Macro F1）。

### English Summary

The paper proposes counterfactual context revision as an auditing framework for LLM-based stance simulation. Starting from 1,821 Reddit conversations about AI models, the authors simulate a target user's stance under the original context, then apply controlled text-only or meme-based multimodal revision strategies to the context, re-simulate the stance, and measure the transition rate and directional shift. Findings reveal that LLM stance simulations are sensitive to semantically independent context changes, raising concerns about the reliability of LLMs as proxies for specific human opinions in social science research and content governance applications.

---

## 关键指标 / Key Metrics

| Setting | Metric | Value |
|---------|--------|-------|
| Original context simulation | Accuracy | **77.64** |
| Original context simulation | Macro F1 | **78.10** |
| Dataset scale | Conversations | **1,821** |
| Context revision | Stance transition rate | Significant (paper-reported) |
