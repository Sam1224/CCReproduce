# OneRetrieval: Unifying Multi-Branch E-commerce Retrieval with an Editable Generative Model

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **Title** | OneRetrieval: Unifying Multi-Branch E-commerce Retrieval with an Editable Generative Model |
| **ArXiv ID** | [2606.13533](https://arxiv.org/abs/2606.13533) |
| **Authors** | Xuxin Zhang, Ben Chen, Yue Lv, Siyuan Wang, Yupeng Li, Yufei Ma, Zihan Liang, Tong Zhao, Ying Yang, Huangyu Dai, Lingtao Mao, Zhipeng Qian, Xinyu Sun, Chenyi Lei, Wenwu Ou, Kun Gai |
| **Affiliation** | Kuaishou Technology (快手) |
| **Submitted** | 2026-06-11 |
| **Source** | HuggingFace June 13 daily listing |
| **Bucket** | STRONG — 电商检索、生成式召回、可编辑/运营干预 |
| **Code** | `2026-06-13/OneRetrieval/` |

---

## 方法概述 / Method Overview

**故事弧线：** 工业电商召回通常是多分支拼接（倒排/稠密/协同/生成式），各分支维护成本高、新词注入需数小时重索引。现有生成式检索方法虽然在端到端质量上有优势，但放弃了倒排召回"小时级运营干预"这一工业核心优势。→ OneRetrieval 通过 Keyword-Aligned Encoding（KAE）让生成式检索同样支持无需重训练的关键词绑定，实现"一模型统一多路召回"同时保留运营可控性。

**核心技术：**
1. **Keyword-Aligned Encoding (KAE)**：将生成式检索的 identifier 每一位置与可解释的属性关键词对齐（而非量化 embedding），使每个 codebook slot 有语义含义；
2. **Reserved Slots**：在每个 codebook 预留 slots，运营可将新出现关键词与目标商品集绑定到 reserved slot，实现无重训练的小时级注入；
3. **信息论式 Codebook 合并**：通过互信息最大化对 codebook 分组，平衡召回质量与多样性；
4. **四阶段 SFT**：渐进式微调确保 identifier 语义对齐与召回指标兼顾。

**工业意义：** 先替换倒排分支（+0.71% Order），再替换稠密分支（Item CTR +0.82%），验证了逐步统一多路召回的可行性。

---

## 关键指标 / Key Metrics

| 评测场景 | 指标 | OneRetrieval | 对比基线 |
|---------|------|-------------|---------|
| 离线（Order 集）| HR@350 | 0.5482 | 最强 generative baseline |
| 离线（Click 集）| HR@350 | 0.6055 | 最强 generative baseline |
| 线上 AB（替换倒排）| Order | **+0.710%** (p<0.05) | 原多路召回 |
| 线上 AB（替换倒排）| Buyer | +0.450% | 原多路召回 |
| 线上 AB（替换几乎全部）| Item CTR | **+0.821%** (p<0.05) | 原多路召回 |

---

## 评分 / Scoring

| 维度 | 得分 | 满分 | 说明 |
|------|------|------|------|
| 方法创新性 | 27 | 30 | KAE + reserved slots 解决工业刚需"新词小时级注入"，创新性强 |
| 实验指标 | 13 | 15 | 线上 AB 业务指标 + 离线召回指标双验证 |
| 实验质量 | 13 | 15 | 多阶段 ablation，工业 A/B 设计严谨 |
| 方法效率 | 6 | 10 | beam-search 推理较重；但运营成本大幅降低 |
| 方法泛化性 | 3 | 5 | 高度依赖电商场景特征 |
| 领域相关性 | 25 | 25 | 核心电商召回 + 可运营干预，直接落地 |
| **Total** | **87** | **100** | |

**复现路径：** `2026-06-13/OneRetrieval/`

---

## Story Arc

> **现状不足：** 生成式检索质量好但牺牲运营干预能力；倒排可运营但质量有限；两者难以统一。  
> **解法：** KAE 将 identifier 位置与关键词语义绑定 → Reserved slots 支持新词无重训练注入 → 四阶段 SFT 保证质量 → 线上 AB 证明逐步替换多路召回可行且有业务收益。
