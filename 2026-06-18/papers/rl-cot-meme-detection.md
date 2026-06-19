# Adapting RL with Chain-of-Thought Supervision for Explainable Detection of Hateful and Propagandistic Memes

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | Adapting Reinforcement Learning with Chain-of-Thought Supervision for Explainable Detection of Hateful and Propagandistic Memes |
| **Authors** | Mohamed Bayan Kmainasi, Mucahid Kutlu, Ali Ezzat Shahroor, Abul Hasnat, Firoj Alam |
| **Affiliation** | (Multi-institution, NLP/Content Moderation group) |
| **arXiv ID** | [2606.15307](https://arxiv.org/abs/2606.15307) |
| **Submitted** | June 14, 2026 |
| **Venue** | Preprint |

---

## 方法概述 / Method Summary

### Story Arc

> **现有方法的问题**：仇恨表情包（meme）通过图像与文字的交互传达有害意图，而单一模态无法揭示。尽管"思考型"多模态大型语言模型（thinking-based MLLM，如o3等）在视觉语言理解上取得进展，其在表情包内容审核中的应用仍缺乏系统探索，且现有模型在分类准确率和解释质量之间存在权衡矛盾。
>
> **解决方案**：提出基于强化学习的后训练方法，通过任务特定奖励和GRPO（Group Relative Policy Optimization）改进thinking-based MLLM的分类性能和参考解释质量，同时引入思考长度正则化防止过度冗长，并探索基于无标注表情包共识伪标签的自监督GRPO。

### Technical Approach (EN)

The paper proposes RL post-training for Multimodal LLMs (MLLMs) specifically for hateful and propagandistic meme detection:

1. **GRPO with Task-specific Rewards**: Group Relative Policy Optimization with rewards that jointly optimize (a) classification accuracy on meme harmfulness labels and (b) quality of chain-of-thought explanations via reference-based scoring.
2. **Thinking-length Regularization**: Prevents degenerate long/short thinking chains; regularizes the length of the model's internal reasoning process.
3. **Semi-supervised Extension**: Self-supervised GRPO using consensus pseudo-labels from multiple model predictions on unlabeled memes — extends the approach to low-resource scenarios.
4. **Distillation-based CoT rationales**: Extends existing meme datasets with weakly supervised chain-of-thought rationales via distillation from stronger teacher models.

Evaluated on English (FHM - Hateful Memes) and Arabic (ArMeme) benchmarks.

### 创新亮点 (ZH)

- **GRPO用于多模态内容审核**：首次将Group Relative Policy Optimization应用于仇恨表情包检测任务，将分类准确率与解释质量纳入统一RL目标。
- **思考长度正则化**：防止thinking-based MLLM产生过长或过短推理链，提高实用性。
- **跨语言验证**：在英语（FHM）和阿拉伯语（ArMeme）数据集上均有提升，展示多语言泛化能力。
- **半监督扩展**：无标注数据上的自监督GRPO降低了标注成本门槛。

---

## 关键指标 / Key Metrics

| Dataset | Metric | This Paper | Best Baseline |
|---------|--------|------------|---------------|
| FHM (English Hateful Memes) | Accuracy | **82.0%** (+2.1%) | 79.9% |
| ArMeme (Arabic Memes) | Macro-F1 (w/ explanation) | **0.612** (+7.6 pts) | 0.536 |
| ArMeme (Arabic Memes) | Macro-F1 (w/o explanation) | +6.1 pts vs ArMeme benchmark | ArMeme benchmark |

---

## 评分详情 / Scoring Breakdown

| Dimension | Sub-score | Justification |
|-----------|-----------|---------------|
| Innovation | 22/30 | GRPO + CoT rewards + length regularization for MLLM is novel; pseudo-label self-supervised extension adds value |
| Experimental SOTA delta | 11/15 | Consistent improvements across both benchmarks; moderate absolute delta |
| Experimental quality / ablations | 12/15 | Systematic ablations on GRPO components, thinking length, self-supervised mode |
| Efficiency | 7/10 | RL post-training adds compute; deployed on existing MLLM backbone |
| Generalization | 4/5 | Cross-language (EN + AR); cross-meme-type (hateful + propagandistic) |
| Domain relevance (ecom+governance) | 18/25 | Strong content moderation relevance; meme detection is adjacent to e-commerce content governance |
| **Total** | **74/100** | |
