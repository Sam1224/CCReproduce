# Beyond Item IDs: Scaling Short-Form-Video Recommendation via Semantic-Native Long Sequence Modeling

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | Beyond Item IDs: Scaling Short-Form-Video Recommendation via Semantic-Native Long Sequence Modeling |
| **Authors** | Ruixiao Sun, Diego Uribe Mora, Zhimeng Jiang, Yuanzhen Lin, Jiarui Wang, Yuening Li, Danfeng Guo, Zhizhong Chen, Chuan He, Liang Liu |
| **Affiliation** | Google, Mountain View, USA |
| **arXiv ID** | [2606.07546](https://arxiv.org/abs/2606.07546) |
| **Submitted** | June 2026 (Google production system) |
| **Venue** | Preprint / Production System |
| **Code** | Not released (Google internal) |
| **Reproduction** | [`code/SemanticNativeSFVRec/`](../../code/SemanticNativeSFVRec/) |

---

## 方法概述 / Method Summary

### Story Arc

> **现有方法的问题**：短视频推荐系统在建模超长用户历史（watch history）时面临两个关键瓶颈：（1）原子Video ID的语义稀疏性——每个视频被随机分配ID，无法编码内容语义；（2）Transformer的二次方计算复杂度限制序列长度。这使得冷启动视频（新创作者内容）在推荐系统中天然处于劣势。
>
> **解决方案**：提出基于语义原生的长序列建模框架，以内容原生的语义ID（Semantic IDs）替代传统原子Video ID，并使用深度截断的粗粒度语义ID缩小Embedding表规模，通过共享语义前缀自然地泛化到冷启动内容。

### Technical Approach (EN)

This paper presents a production-deployed framework at Google (billion-user scale) addressing long user behavior sequence modeling for short-form video recommendation. Key innovations:

1. **Semantic IDs instead of atomic IDs**: Replace randomly assigned Video IDs with content-derived Semantic IDs generated via hierarchical quantization of video embeddings. These IDs encode content semantics and share prefixes across similar content, naturally solving the cold-start problem.
2. **Depth-truncated coarse-grained Semantic IDs**: Use only the first few levels of the semantic ID hierarchy (coarse-grained), reducing the embedding table from corpus cardinality (billions of videos) to a manageable size while preserving semantic structure.
3. **Ultra-long sequence modeling**: With smaller embedding tables and semantic coherence, sequences can be scaled to much longer histories without the quadratic complexity exploding.

The framework is deployed in production at Google serving billions of users.

### 创新亮点 (ZH)

- **语义原生表征**：用内容语义ID替代随机Video ID，彻底解决ID稀疏和语义断层问题。
- **深度截断的粗粒度设计**：仅使用层次量化的前几层，大幅压缩Embedding表规模（从数十亿降至可控量级）。
- **冷启动自然泛化**：新视频通过共享语义前缀被纳入已有序列空间，无需专门冷启动处理。
- **十亿用户级生产部署**：在Google真实系统中验证，具有极高工业价值。

---

## 关键指标 / Key Metrics

| Dataset / Setting | Metric | Result |
|-------------------|--------|--------|
| Google short-form video (production) | Online engagement (A/B) | Statistically significant gains vs. item-ID baseline |
| Cold-start videos | Watch rate | Improved via semantic prefix sharing |
| Embedding table size | Compression ratio | Corpus-cardinality → manageable fixed size |
| Sequence length supported | Max tokens | Scales beyond prior Transformer limits |

---

## 评分详情 / Scoring Breakdown

| Dimension | Sub-score | Justification |
|-----------|-----------|---------------|
| Innovation | 22/30 | Semantic IDs for sequence modeling is known; key contribution is depth-truncated coarse-grained design + cold-start generalization at scale |
| Experimental SOTA delta | 12/15 | Production A/B at Google billion-user scale; highly credible but no public ablation numbers |
| Experimental quality / ablations | 11/15 | Deployed system paper; limited academic ablations |
| Efficiency | 9/10 | Production deployed, explicitly optimizes for compute efficiency |
| Generalization | 4/5 | Naturally generalizes to cold-start; platform-agnostic principle |
| Domain relevance (ecom+governance) | 22/25 | Short-form video (influencer 达人 content platform); content embedding; recommendation — high relevance |
| **Total** | **80/100** | |

---

## 差异化分析 / Novelty vs. Prior Work

| Prior Work | Gap | This Paper |
|---|---|---|
| BERT4Rec / SASRec | Atomic IDs, no semantics | Content-native Semantic IDs |
| RQ-VAE Semantic IDs (TIGER) | Full-depth, large tables | Depth-truncated, compact, cold-start friendly |
| Long sequence work (ETA, SIM) | Sampling over long IDs | Semantic prefix shrinks table; enables longer sequences |
