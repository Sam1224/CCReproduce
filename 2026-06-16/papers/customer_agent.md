# Customer-Agent: Ultra-Long Shopping Trajectories

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **标题** | Customer-Agent: Overcoming Context Limitations in Ultra-Long Shopping Trajectories via Tool-Augmented Agents and RLVR |
| **作者** | Hongye Liu, Rongmei Lin, Anurag Kashyap, Hejie Cui, Ricardo Henao, Besnik Fetahu, Bing Yin |
| **机构** | Amazon, Duke University |
| **链接** | https://arxiv.org/abs/2606.07995 |
| **arXiv ID** | 2606.07995 |
| **提交日期** | June ~7, 2026 |
| **Bucket** | STRONG |

---

## 方法概述 / Method Overview

**中文：**  
来自Amazon和Duke大学的工作，针对电商购物历史轨迹极长（跨越数年数千条购买记录）导致LLM上下文窗口溢出的问题，提出了Customer-Agent框架。系统将用户购物轨迹存储为外部本地文件，训练LLM智能体通过代码解释器（code-interpreter）自主调用工具进行检索和解析。引入ShopTrajQA基准，使用RLVR（可验证奖励强化学习）训练智能体在长轨迹推理任务上的能力。

**English:**  
Amazon/Duke propose Customer-Agent to handle ultra-long shopping trajectories that exceed LLM context windows. Shopping histories (spanning years, thousands of purchases) are stored as external files; the LLM agent retrieves and parses them via code-interpreter tool calls. A new ShopTrajQA benchmark enables RLVR training for long-trajectory reasoning tasks.

---

## 故事弧线 / Story Arc

**现有方法不足 →** 用户多年购物历史动辄数千条，远超LLM上下文窗口；截断会损失关键历史信息。  
**本文设计 →** 工具增强智能体将历史轨迹外化为文件，通过自主代码调用精准检索相关片段，RLVR训练保证推理准确性。

---

## 创新性 / Innovation

1. **购物轨迹外化存储+工具调用**：突破LLM上下文限制，利用代码解释器动态检索所需轨迹片段。
2. **ShopTrajQA基准**：首个超长购物轨迹问答评估基准，基于真实商品信息和模拟轨迹构建。
3. **RLVR训练范式**：可验证奖励强化学习用于电商轨迹推理，相比SFT更可靠。

---

## 关键指标 / Key Metrics

| 数据集 | 指标 | 结果 |
|--------|------|------|
| ShopTrajQA | 长轨迹推理准确率 | 优于标准LLM |
| ShopTrajQA | vs. RAG基线 | 工具调用方式更准确 |

---

## 评分 / Scoring

| 维度 | 分数 | 满分 | 说明 |
|------|------|------|------|
| 创新性 (Innovation) | 20 | 30 | 工具增强轨迹外化，实用创新 |
| 实验SOTA增量 (SOTA Delta) | 10 | 15 | ShopTrajQA上有提升 |
| 实验质量/消融 (Exp Quality) | 11 | 15 | 新基准+对比实验完整 |
| 效率 (Efficiency) | 7 | 10 | 工具调用引入额外开销 |
| 泛化性 (Generalization) | 3 | 5 | 主要针对购物历史场景 |
| 领域相关性 (Domain Relevance) | 20 | 25 | 电商用户行为理解，直接相关 |
| **Total** | **71** | **100** | |

---

## 参考链接

- arXiv: https://arxiv.org/abs/2606.07995
