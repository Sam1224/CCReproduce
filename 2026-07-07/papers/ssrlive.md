# SSRLive: Live Streaming Recommendation with Dynamic Semantic ID

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **标题** | SSRLive: Live Streaming Recommendation with Dynamic Semantic ID |
| **作者** | Teng Shi, Zhaoheng Li, Yuanhang Qu, Yi Liu, Lixiang Lai, Yuning Jiang |
| **机构** | Taobao & Tmall Group, Alibaba |
| **arXiv** | https://arxiv.org/abs/2606.06970 |
| **发表日期** | 2026-06-07 |
| **标签** | Live Streaming · Semantic ID · Recommendation · Generative+Discriminative · Alibaba |

---

## 故事弧 / Story Arc

**现有困境：** 直播推荐中 item 表示难以建模——直播内容随时间动态变化（节目切换、促销信息更新、主播状态变化），静态嵌入或 ID 无法捕捉这种实时性；行为数据稀疏（冷启）使传统协同过滤效果有限。

**我们的设计：** SSRLive 引入动态语义 ID（Dynamic Semantic ID），在统一生成-判别框架中实时刻画直播内容状态，解决静态表示的时效性缺陷。

---

## 方法摘要 / Method Overview

### 统一生成-判别框架

**生成组件（Static + Dynamic SID 生成）**
- 编码器-解码器结构
- 静态 SID：编码直播间的持续属性（主播风格、历史内容主题）
- 动态 SID：编码当前实时状态（当前节目内容、实时用户反应、促销信息）
- 多模态输入：视频帧、音频、文字信息

**判别组件（用户匹配预测）**
- 输入：静态 SID + 动态 SID + 用户特征
- 主播-用户历史交互数据增强
- 多任务预测（点击率、观看时长、互动量等）

**端到端联合训练**：生成模块与判别模块共享表示，协同优化。

---

## 关键指标 / Key Metrics

| 指标 | 线上 A/B 提升 |
|------|--------------|
| 观看时长 | **+3.38%** |
| GMV（成交额） | **+0.72%** |
| 新增粉丝 | **+3.12%** |
| 互动量 | **+2.92%** |

> 全量上线，服务数亿活跃用户；离线实验在生产规模数据集上持续超越所有竞争基线。

---

## 评分 / Score

| 维度 | 得分 | 最高 |
|------|------|------|
| Innovation | 19 | 30 |
| Experimental SOTA delta | 12 | 15 |
| Experimental quality / ablations | 11 | 15 |
| Efficiency | 7 | 10 |
| Generalization | 3 | 5 |
| Domain relevance (ecom + governance) | 22 | 25 |
| **Total** | **74** | **100** |

**评分理由：** 阿里巴巴直播电商核心推荐技术，动态语义 ID 解决真实痛点，GMV 提升直接体现商业价值；全量上线验证工程可行性。扣分原因：生成-判别统一框架独创性中等，泛化讨论有限。
