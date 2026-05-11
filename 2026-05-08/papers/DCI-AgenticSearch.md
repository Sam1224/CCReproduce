# Beyond Semantic Similarity: Rethinking Retrieval for Agentic Search via Direct Corpus Interaction

- **arXiv**: https://arxiv.org/abs/2605.05242
- **HuggingFace 关注度**: 37 upvotes
- **作者**: Zhuofeng Li, Haoxiang Zhang, Cong Wei, Pan Lu, Ping Nie, Yi Lu 等 (TIGER-Lab)
- **领域标签**: `Agentic Search` `Retrieval` `RAG` `Tools` `电商-检索`

## 方法概述
**Direct Corpus Interaction (DCI)** — 让 agent 直接用 grep / 文件读 / shell 命令 / 轻量脚本访问原始语料，**完全不走 embedding/向量索引/检索 API**。论文把现代检索抽象成"固定相似度接口下的 top-k 压缩"，认为对 agentic 任务这是瓶颈：精确词约束、稀疏线索合取、局部上下文检查、多步假设修正都难用 off-the-shelf retriever 做。

## 故事
检索固定接口 → 一次 top-k 把信息压扁 → agent 在多跳证据/弱线索/精确约束下"被动失血" → 直接让 agent 用 terminal 工具读原始语料，不要 embedding。

## 创新性分析
- **接口设计层面的反思**: 不是发明新 retriever，而是质疑"retriever 该不该存在"。
- 把 agentic search 的瓶颈定位到"接口分辨率"，而非推理能力，是一个能引发后续工作的论点。
- 工程实现门槛低：grep + shell；与 LLM agent 框架天然契合。
- 与 ReAct / Toolformer 类工作正交：他们说 agent 用工具，DCI 说 agent 用"通用 unix 工具直接访问 corpus"。

## 关键指标
- 在 BRIGHT、BEIR 多个子集上 substantially outperforms sparse/dense/rerank 强 baseline；
- 在 BrowseComp-Plus、multi-hop QA 上"高准确率"，不依赖任何 semantic retriever。
- 摘要未给具体数字。

## 评分
| 维度 | 分 / 满分 | 理由 |
|------|----------|------|
| 创新性 | 25 / 30 | 接口层反思具突破性 |
| 实验指标 | 11 / 15 | 多 benchmark 报告但缺具体百分点 |
| 实验质量 | 11 / 15 | benchmark 广度合理 |
| 效率 | 6 / 10 | 无离线索引是 +；推理时反复 grep 大语料是 - |
| 泛化性 | 4 / 5 | 通用 corpus 适配 |
| 相关性 | 18 / 25 | 电商 customer agent / GMP / 商家工作台 agent 直接受益 |
| **合计** | **75** |
