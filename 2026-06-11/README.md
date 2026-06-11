# 2026-06-11 电商内容生态 & 达人治理 — Paper 巡检

本日巡检窗口：**GMT+8 2026-06-11 00:00–23:59**（对应 **UTC 2026-06-10 16:00–2026-06-11 15:59**）。主要从 arXiv（cs.IR/cs.CL/cs.CV/cs.AI/cs.LG/stat.ML）拉取近 24h 更新，并按“电商内容理解/推荐/检索/去重/治理 + LLM/VLM/RAG/数据”关键词二次筛选，跳过安全/后门方向论文。

本日共整理 7 篇候选论文，评分与摘要见 `papers.json`。

## 建议关键词池（可每日固定订阅 + 二次过滤）

强相关（电商内容生态&达人治理）：recommendation / recsys、cold-start、ranking / rerank、retrieval / vector database、dedup / near-duplicate、content understanding、caption / multimodal、clustering / similarity、violation detection / moderation、治理 / 审核 / 质量、data cleaning / dedup / labeling。

弱相关但值得关注（大模型方向）：LLM/VLM/MLLM、RAG/agent、KV-cache / long-context、token pruning / routing、distillation、efficient inference。

## 推荐补充的 paper sources（便于每日抓新）

- 论文聚合：HuggingFace Daily Papers、Papers with Code（Trending / Latest）、arXiv-sanity、Connected Papers。
- arXiv 入口：固定订阅 cs.IR/cs.CL/cs.CV/cs.LG/cs.AI 的 daily listing（或 RSS），再用关键词二次过滤（尤其是 recsys / dedup / RAG / moderation）。
- 会议/评审：OpenReview（ICLR/NeurIPS/ICML/ACL 等）按「recent submissions / decisions」；会议 accepted list 页面（CVPR/ICCV/ECCV、KDD/SIGIR、SIGMOD/VLDB）。
- 学术检索：Semantic Scholar（Latest + topic alerts）、Google Scholar Alerts、OpenAlex（concept + date filter）、DBLP（会议/期刊增量）。
- 工业界产出：Google Research/DeepMind、Meta AI、Microsoft Research、Amazon Science、NVIDIA Research、ByteDance Seed、Qwen、Hunyuan 等官方 blog/RSS。

## 评分机制（100 分）

- 方法创新性：30
- 实验指标：15
- 实验质量：15
- 方法效率：10
- 方法泛化性：5
- 论文相关性：25

## 80+ 高质量论文复现

- DiffCold（2606.12245）：已在 `2026-06-11/DiffCold` 提供可运行 toy pipeline（数据/模型/训练/测试），复现“检索增强初始化 + 条件扩散生成 cold item embedding + 对齐损失”的关键思想。
- MLT-Dedup（2606.12215）：已在 `2026-06-11/MLT-Dedup` 提供可运行 toy pipeline（合成视频序列数据、两级表示检索、差分特征匹配与时序重叠定位）。
- LLM-Based User Personas（2606.12198）：已在 `2026-06-11/LLMUserPersonas` 提供工程化原型（用户历史聚类、persona 生成接口/异步服务骨架、基于 persona 的候选召回示例；LLM 生成与蒸馏部分以可替换模块形式提供）。

> 说明：以上复现以“**关键机制可跑通 + 最小闭环**”为目标，使用合成/简化数据验证核心思想；与论文的真实数据、训练规模与线上系统仍有差距。