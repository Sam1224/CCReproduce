# TGQ-Former: Text-Guided Visual Representation Learning for Robust Multimodal E-Commerce Recommendation

## 基本信息 / Basic Info

| Field | Value |
|-------|-------|
| **Title** | Text-Guided Visual Representation Learning for Robust Multimodal E-Commerce Recommendation |
| **arXiv** | [2605.17366](https://arxiv.org/abs/2605.17366) |
| **Authors** | Yufei Guo, Jing Ma, Tianlu Zhang, Shijie Yang, Yanlong Zang, Weijie Ding, Pinghua Gong, Jungong Han |
| **Affiliations** | Tsinghua University, JD.COM |
| **Date** | 2026-05-17 |
| **Bucket** | STRONG |
| **Total** | **76 / 100** |

---

## 故事弧 / Story Arc

> **问题:** 电商商品图片通常包含促销叠加文字、水印、背景干扰等噪声视觉信息，这些"虚假视觉线索"通过视觉编码器注入商品嵌入，严重降低I2I（商品到商品）检索的鲁棒性。在MLLM风格的检索流水线（冻结视觉编码器 + 轻量连接器 + LLM）中，连接器必须从视觉token序列中选择性聚合有意义的语义信息。
>
> **方案:** TGQ-Former（Text-Guided Q-Former）：将结构化商品元数据（标题、类目、属性等）作为语义引导，驱动Q-Former从视觉token中选取与商品语义相关的token，过滤促销噪声。
>
> **差异:** 传统Q-Former用固定可学习查询token，无法感知具体商品语义。TGQ-Former将文本元数据动态编码为查询引导，实现语义感知的视觉token选择，在保留互补视觉证据的同时抑制噪声。

---

## 方法概述 / Method Summary

**架构流程:**

```
Product Image → Frozen Vision Encoder → Visual Tokens
                                              ↓
Product Metadata (title, category, attrs) → Text Encoder → Semantic Anchors
                                              ↓
                                    TGQ-Former Connector
                                    (Text-guided Q-Former)
                                              ↓
                               Filtered Visual Tokens → LLM → Item Embedding
```

**核心组件:**

1. **文本语义锚点生成 (Semantic Anchor Generation):**
   - 将商品结构化元数据（标题+类目+属性）送入文本编码器
   - 生成语义锚点向量，捕获商品的核心语义属性

2. **TGQ-Former视觉token选择:**
   - 以语义锚点作为Q-Former的初始化查询（替代随机初始化）
   - Cross-attention机制从视觉token中选择与文本语义对齐的视觉token
   - 抑制与促销文字/背景对应的噪声视觉token

3. **互补视觉证据保留:**
   - 保留语义锚点未覆盖但视觉上唯一的商品特征（颜色、纹理、材质）
   - 平衡文本引导与视觉多样性

**训练目标:**

$$\mathcal{L} = \mathcal{L}_{I2I} + \alpha \cdot \mathcal{L}_{align}$$

其中 $\mathcal{L}_{I2I}$ 是对比检索损失，$\mathcal{L}_{align}$ 是视觉-文本对齐损失。

---

## 创新性分析 / Innovation

1. **工业问题导向:** 精确识别并解决了电商场景特有的视觉噪声问题（促销叠加），具有极强实用价值
2. **元数据利用新思路:** 将商品结构化元数据从辅助信息提升为视觉特征提取的引导信号
3. **可行性强:** 只需修改MLLM的连接器部分，与任意冻结视觉编码器和LLM骨干兼容
4. **JD.COM真实部署背景:** 实验在真实电商数据上进行，结论具有工业可信度

---

## 关键指标 / Key Metrics

| Task | Dataset | Metric | TGQ-Former | Baseline |
|------|---------|--------|------------|----------|
| I2I Retrieval (noisy images) | JD.COM product data | Recall@K | +↑ | LLaVA connector |
| I2I Retrieval (clean images) | JD.COM product data | Recall@K | 持平/略升 | baseline |
| Robustness (w/ promotional overlays) | JD.COM | Recall@K | 显著提升 | frozen ViT |

---

## 评分详情 / Score Breakdown

| Dimension | Score | Max | Justification |
|-----------|-------|-----|---------------|
| Innovation | 22 | 30 | 元数据引导视觉token选择，针对电商噪声的精准设计 |
| Experimental SOTA delta | 11 | 15 | 在有促销噪声图像上显著提升；有JD.COM真实数据支持 |
| Experimental quality / ablations | 11 | 15 | 消融验证引导机制；干净/噪声数据对比实验充分 |
| Efficiency | 7 | 10 | 只修改连接器，不增加推理overhead |
| Generalization | 3 | 5 | 验证于电商场景，其他场景迁移有限 |
| Domain relevance (ecom + governance) | 22 | 25 | 直接解决电商I2I检索核心问题，来自JD.COM工业实践 |
| **Total** | **76** | **100** | — |

---

## 与本领域关联 / Domain Relevance

- **电商相似商品推荐:** 直接提升I2I检索质量，影响商品推荐核心链路
- **商品图片质量治理:** 促销叠加检测/过滤是内容生态治理的重要环节
- **达人选品辅助:** 高质量商品嵌入支持达人/KOL商品匹配
- **多模态商品理解:** JD.COM工业验证，可直接迁移至淘宝/拼多多等场景
