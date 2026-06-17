# LLM-Based User Personas for Recommendations at Scale

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **标题** | LLM-Based User Personas for Recommendations at Scale |
| **作者** | Haoting Wang, Haokai Lu, Zheyun Feng, Jenny Huang, Yifat Amir, Gregory Hinkson, Ben Most, Zelong Zhao, Yixin Kelly Cui, Rein Zhang, Fabio Soldo, Yu Xia, Nihar Bhupalam, Minmin Chen, Konstantina Christakopoulou, Lichan Hong, Ed H. Chi |
| **机构** | Google |
| **链接** | https://arxiv.org/abs/2606.12198 |
| **arXiv ID** | 2606.12198 |
| **提交日期** | June 10, 2026 |
| **Bucket** | STRONG |

---

## 方法概述 / Method Overview

**中文：**  
本文来自Google，提出了一个在十亿级用户规模的商业视频推荐平台上实时生成LLM用户兴趣画像（User Interest Personas）的完整框架。核心方法通过大型语言模型生成自然语言用户兴趣描述，解决兴趣探索-利用（exploration-exploitation）的均衡问题——不仅摘要既有兴趣，还能生成新话题引导探索。为克服在线LLM推理的计算瓶颈，设计了基于**语义聚类视频表示**的输入优化、**知识蒸馏（KD）**和**异步推理**三重成本控制机制。

**English:**  
Google proposes a framework for real-time LLM user interest persona generation on a billion-user commercial video recommendation platform. The LLM generates natural-language user interest personas that balance exploitation (summarizing known interests) with exploration (novel topic generation). Three cost-reduction mechanisms enable billion-scale deployment: semantically clustered video representations for input compression, knowledge distillation to a smaller serving model, and asynchronous inference.

---

## 故事弧线 / Story Arc

**现有方法不足 →** 结构化ID或离线处理缺乏语义丰富性和实时适应性；用户兴趣无法以可解释形式呈现。  
**本文设计 →** 在线生成自然语言兴趣画像 + KD压缩 + 异步异步推理 + 语义聚类压缩输入，实现十亿级在线部署。

---

## 创新性 / Innovation

1. **在线LLM推理用于推荐**：首次在十亿级规模系统中实现在线LLM推理，而非离线预计算。
2. **语义聚类视频表示**：将用户历史视频语义聚类为紧凑表示，大幅降低LLM输入长度。
3. **探索-利用均衡画像**：LLM生成的画像同时包含兴趣摘要和新话题探索，改善长期用户留存。
4. **知识蒸馏+异步推理**：工程级创新，使高计算成本的LLM推理在在线serving中可行。

---

## 关键指标 / Key Metrics

| 场景 | 指标 | 结果 |
|------|------|------|
| 视频推荐平台 (Google) | 在线A/B CTR/时长提升 | 显著提升（具体数值未公开） |
| 系统性能 | 推理延迟控制 | 满足在线serving要求 |
| 知识蒸馏 | 小模型vs教师模型质量差距 | 接近教师模型 |

---

## 评分 / Scoring

| 维度 | 分数 | 满分 | 说明 |
|------|------|------|------|
| 创新性 (Innovation) | 22 | 30 | 在线LLM画像+三重效率机制 |
| 实验SOTA增量 (SOTA Delta) | 10 | 15 | 生产指标有提升但未公开详细数值 |
| 实验质量/消融 (Exp Quality) | 12 | 15 | Google生产环境+消融实验 |
| 效率 (Efficiency) | 9 | 10 | 专为十亿级在线推理设计 |
| 泛化性 (Generalization) | 4 | 5 | 可迁移至视频/内容推荐系统 |
| 领域相关性 (Domain Relevance) | 20 | 25 | 内容推荐系统，与电商达人内容分发高度相关 |
| **Total** | **77** | **100** | |

---

## 参考链接

- arXiv: https://arxiv.org/abs/2606.12198  
- X (Twitter): https://x.com/_reachsumit/status/2064950642171617329
