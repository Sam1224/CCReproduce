# EKTM: Effective Knowledge Transfer for Multi-Task Recommendation Models

- **arXiv**: https://arxiv.org/abs/2605.05730
- **作者**: Guohao Cai, Jun Yuan, Zhenhua Dong (Huawei Noah's Ark Lab 风格作者署名)
- **领域标签**: `MTL` `CVR Prediction` `Recommendation` `Knowledge Transfer` `电商核心`

## 方法概述
EKTM 解决工业 CVR 预估的稀疏性问题：用一个 **router** 模块把跨任务 (CTR / 加购 / 收藏 / CVR) 的知识聚合分发；每个 CVR 任务挂一个 **transmitter** 把 router 产出转换到本任务空间；再加一个 **enhanced** 模块保证迁移后的特征不损害本任务原始学习。

## 故事
CVR 标签稀疏 → 直接训练困难 → 借同源行为 (CTR / 加购等) 做知识迁移 → 但 naive 共享会负迁移 → router/transmitter/enhanced 三件套显式控制信号流向。

## 创新性分析
- 三模块分工是 PLE / MMoE / STAR 思路的延展，**架构级新颖度有限**；
- 但解决 negative transfer 的工程动机和落地可行性强；
- 在 CVR 上做 MTL 知识迁移，是电商场景非常实用的方向。

## 关键指标
- 商业平台 A/B：**eCPM +3.93%**。
- 离线指标：在 benchmark 数据集报告但摘要未给具体数字。

## 评分
| 维度 | 分 / 满分 | 理由 |
|------|----------|------|
| 创新性 | 18 / 30 | 模块组合，相比 PLE 增量 |
| 实验指标 | 11 / 15 | 在线 +3.93% eCPM |
| 实验质量 | 11 / 15 | 商业 A/B 是强证据 |
| 效率 | 6 / 10 | 模块级开销，可接受 |
| 泛化性 | 3 / 5 | 与平台耦合 |
| 相关性 | 22 / 25 | CVR 是电商付费链路核心 |
| **合计** | **71** |
