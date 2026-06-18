# Detecting AI-Generated Content on Social Media with Multi-modal Language Models

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | Detecting AI-Generated Content on Social Media with Multi-modal Language Models |
| **Authors** | Chenyang Yang, Shen Yan, Yibo Yang, Litao Hu, Yuchen Liu, Yuan Zeng, Hanchao Yu, Yinan Zhu, Sumedha Singla, Brian Vanover, Huijun Qian, Zihao Wang, Fujun Liu, Aashu Singh, Jianyu Wang, Xuewen Zhang |
| **Affiliations** | Carnegie Mellon University; Meta |
| **arXiv** | [2606.11200](https://arxiv.org/abs/2606.11200) |
| **Submitted** | ~2026-06-09 |
| **Domain Tags** | AIGC detection, content governance, multimodal, social media, Meta |

---

## 方法概述 / Method Summary

生成式 AI 已能创作以假乱真的图像和视频，并被大量用于社交媒体上的垃圾信息、虚假信息、操控和欺诈。现有 AIGC 检测方法面临：对新生成模型泛化差、依赖单一模态、缺乏可解释输出三大痛点。本文提出一套端到端的多模态 AIGC 检测流水线：通过**持续策划**多样化多模态社交媒体数据（含图文视频跨平台），训练一个紧凑型视觉-语言模型（compact VLM）同时完成检测与解释。模型已在 Meta 平台实际部署，并对帖子推荐产生正面影响（提升用户互动）。

**Story arc**: 平台上的 AI 生成内容（AIGC）治理滞后于生成技术的进步，单模态检测泛化弱、无法解释 → 持续多模态数据策划 + 紧凑 VLM，兼顾检测精度、跨平台泛化与可解释性，且工业落地在 Meta。

**Key components**:
1. **Continuous Data Curation**: 跨平台（Instagram, Facebook, Threads 等）持续采集多模态 AIGC 样本，保持数据多样性
2. **Compact VLM for Detection + Explanation**: 轻量 VLM 同时输出"是否 AIGC"判断和文字解释
3. **Multi-modal Fusion**: 图像、视频帧、文本联合建模
4. **Production Deployment**: 部署于 Meta 帖子推荐系统，正向影响用户互动

---

## 创新性分析 / Innovation Analysis

**vs. prior work**:
- 相比图像 AIGC 检测（如 DIRE、UniversalFakeDetect）单模态限制，本方法支持图文视频多模态
- 持续数据策划机制（Continuous Curation）允许跟踪新生成模型，不需要静态测试集
- Meta 规模实际部署，验证工业可行性
- 解释输出对内容治理审核员有实际价值

**Novelty assessment**: 持续数据策划 + 紧凑 VLM + 解释生成的组合在业界有实际价值，Meta 背书可信。

---

## 关键指标 / Key Metrics

| Dataset/System | Metric | Model | Baseline |
|---------------|--------|-------|----------|
| Public AIGC benchmarks | Detection accuracy | **SOTA** | — |
| Internal Meta datasets | Detection + Explanation | Robust | — |
| Meta deployment | User engagement impact | **Positive** | — |

---

## 评分 / Scoring

| 维度 | 分数 | 满分 | 说明 |
|------|------|------|------|
| Innovation | 22 | 30 | 持续数据策划 + VLM 检测解释，工程为主 |
| Experimental SOTA delta | 11 | 15 | 公开基准 SOTA，内部数据验证 |
| Experimental quality / ablations | 12 | 15 | Meta 规模部署验证 |
| Efficiency | 8 | 10 | 紧凑 VLM，轻量部署 |
| Generalization | 4 | 5 | 跨平台、跨模型泛化设计 |
| Domain relevance | 20 | 25 | AIGC 内容治理直接相关，平台合规核心需求 |
| **Total** | **77** | **100** | Meta 工业部署背书，AIGC 内容治理核心论文 |
