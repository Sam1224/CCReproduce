# LiveStarPro: Proactive Streaming Video Understanding with Hierarchical Memory for Long-Horizon Streams

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | LiveStarPro: Proactive Streaming Video Understanding with Hierarchical Memory for Long-Horizon Streams |
| **Authors** | Zhenyu Yang, Kairui Zhang, Bing Wang, Shengsheng Qian, Changsheng Xu |
| **Affiliations** | Institute of Automation, Chinese Academy of Sciences |
| **arXiv** | [2606.17798](https://arxiv.org/abs/2606.17798) |
| **Submitted** | 2026-06-16 |
| **Domain Tags** | livestream, video understanding, MLLM, hierarchical memory, proactive response |
| **Code** | `code/LiveStarPro/` |

---

## 方法概述 / Method Summary

现有视频大语言模型（Video-LLM）在在线直播场景下面临三重困境：无法持续处理实时视频流、缺乏自主判断回复时机的机制、长时交互中历史信息严重遗忘。LiveStarPro 针对这三个痛点分别提出：**SVeD**（Streaming Verification Decoding）通过单次困惑度验证自动判断回复时机；**SCAM**（Streaming Causal Attention Masks）训练策略实现增量视频-语言对齐；**TSHM**（Tree-Structured Hierarchical Memory）将驱逐的历史信息递归组织为事件链，支持对实际无界视频流的高效检索。三者协同使模型在长时直播理解中同时具备实时响应、主动判断与长程记忆能力。

**Story arc**: 直播内容理解（电商直播合规审核、精彩片段识别）需要在线、实时、长程的视频理解能力，但现有 Video-LLM 只能处理固定长度离线视频 → 设计三组件协同的在线直播理解系统，实现主动式、持续性、层次化记忆的实时直播理解。

**Key components**:
1. **SVeD (Streaming Verification Decoding)**: 通过单次前向 perplexity 验证信号自动识别"应答时机"，无需显式 silence token 监督
2. **SCAM (Streaming Causal Attention Masks)**: 训练时模拟在线流式接收，强制模型学习增量对齐
3. **TSHM (Tree-Structured Hierarchical Memory)**: 树形递归记忆，将长历史流压缩为事件链节点，KV cache 管理高效
4. **Streaming KV Cache**: 1.58× 推理加速，减少显存占用

---

## 创新性分析 / Innovation Analysis

**vs. prior work**:
- 相比 LiveStar（2511.05299，其前作），LiveStarPro 增加了 TSHM 长程记忆和 SVeD 主动回复机制，适用于更长时间跨度
- 相比 HERMES（KV Cache 层次记忆），LiveStarPro 的树形结构支持任意长度事件链检索
- 首次将"主动式回复时机判断"系统性纳入流式视频 LLM 训练框架
- 直接适用于电商直播合规监控：异常内容检测、促销话术识别、违规行为预警

**Novelty assessment**: 三组件设计合理、互补性强；+28.9% 语义正确率的增益显著；1.58× 加速对工业部署友好。

---

## 关键指标 / Key Metrics

| Dataset/Benchmark | Metric | LiveStarPro | Prior SOTA |
|-------------------|--------|-------------|------------|
| Online Video-LLM benchmarks | Semantic Correctness | **+28.9%** | — |
| Online Video-LLM benchmarks | Timing Error | **-18.2%** | — |
| Inference | Speedup | **1.58×** | 1× |

---

## 评分 / Scoring

| 维度 | 分数 | 满分 | 说明 |
|------|------|------|------|
| Innovation | 24 | 30 | 三组件协同（SVeD+SCAM+TSHM），主动式直播理解新范式 |
| Experimental SOTA delta | 13 | 15 | +28.9% 语义正确率，-18.2% 时机误差 |
| Experimental quality / ablations | 12 | 15 | 组件消融实验，但偏学术基准 |
| Efficiency | 9 | 10 | 1.58× 加速，流式 KV 缓存 |
| Generalization | 4 | 5 | 适用各类直播流，泛化性好 |
| Domain relevance | 18 | 25 | 直播视频理解直接适用电商直播内容治理/审核 |
| **Total** | **80** | **100** | 直播内容理解核心论文，达到复现阈值 |
