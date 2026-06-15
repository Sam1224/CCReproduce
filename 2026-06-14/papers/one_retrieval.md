## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **标题** | OneRetrieval: Unifying Multi-Branch E-commerce Retrieval with an Editable Generative Model |
| **作者** | Xuxin Zhang, Ben Chen, Yue Lv, Siyuan Wang, Yupeng Li, Yufei Ma, Zihan Liang, Tong Zhao, Ying Yang, Huangyu Dai, Lingtao Mao, Zhipeng Qian, Xinyu Sun, Chenyi Lei, Wenwu Ou, Kun Gai |
| **机构** | Kuaishou Technology |
| **arXiv** | [2606.13533](https://arxiv.org/abs/2606.13533) |
| **代码** | [github.com/xuxinzhang/oneretrieval](https://github.com/xuxinzhang/oneretrieval) |
| **发布日期** | 2026-06-12 |
| **领域标签** | 电商检索、生成式检索、可运营编辑、Keyword-Aligned Encoding |

---

## 方法概述

**Story Arc**：工业召回通常依赖多路分支（倒排可运营干预但转化弱、稠密/生成式召回质量好但难以小时级注入新词）。本文把"可编辑性"作为 generative retrieval 的结构性目标，通过三项核心设计把两者优势合并：(1) **Keyword-Aligned Encoding (KAE)**：把 identifier 每一位绑定到可解释属性词而非量化 embedding，让 codebook 槽位具备语义可读性；(2) **Reserved Slots**：在 codebook 中预留扩展空间，运营人员可在部署后将新词显式绑定到 reserved slot，实现"无需 retrain 的新词注入"；(3) **四阶段 SFT 流水线**：信息论属性分组 + 分阶段监督微调，把深召回质量做上来。

**Innovation vs Prior Work**：以往 generative retrieval（如 DSI、NCI）依赖模型泛化处理新词，无法满足电商"小时级运营干预"需求；本文把可编辑性作为 identifier 设计的一等约束，并在生产 A/B 测试中验证了"先替换倒排、再逐步替换更多召回分支"的渐进部署路径。

---

## 关键指标

| 数据集 | 指标 | OneRetrieval | 最强 Baseline |
|--------|------|-------------|--------------|
| 快手搜索日志（day31 test） | Order HR@350 | 0.5482 | < OneRetrieval |
| 快手搜索日志（day31 test） | Click HR@350 | 0.6055 | < OneRetrieval |
| 线上 A/B（out-of-mall search） | Order volume | 显著提升 | — |
| 线上 A/B（全召回替换） | Item CTR | 显著提升 | — |

---

## 评分

| 维度 | 得分 | 说明 |
|------|------|------|
| 方法创新性 | 27/30 | "可编辑性" 作为 generative retrieval 结构约束，KAE + reserved slot 是新概念 |
| 实验指标 | 13/15 | 离线检索指标 + 线上 A/B 均有体现，但具体数值部分以定性描述为主 |
| 实验质量 | 13/15 | 31 天真实日志评测，消融覆盖 KAE/reserved slot/staged SFT |
| 效率 | 6/10 | 生成式检索 beam search 开销较重，系统层仍需工程支撑 |
| 泛化性 | 3/5 | 工业场景验证充分，跨平台泛化性待验证 |
| 领域相关性 | 25/25 | 直接解决电商多路召回的"可运营干预 + 深召回"刚需 |
| **Total** | **87/100** | 强电商检索相关，工业刚需直接命中，离线+线上双证据 |

---

## 复现

官方代码：[github.com/xuxinzhang/oneretrieval](https://github.com/xuxinzhang/oneretrieval) — 已验证为非空仓库。  
本仓库路径：`code/OneRetrieval/` → `2026-06-14/OneRetrieval/`（含 verify_repo.py）
