# 2026-06-13 电商内容生态 & 达人治理 — Paper 巡检

本日巡检窗口：**GMT+8 2026-06-13 00:00–23:59**（对应 **UTC 2026-06-12 16:00–2026-06-13 15:59**）。主要以 arXiv（SubmittedDate）为主源，结合“电商内容理解/检索/对话购物助手/治理 + LLM/VLM/Agent/RAG/数据”关键词做二次筛选，跳过安全/后门相关论文。

本日共整理 7 篇候选论文，评分与中英文摘要见 `papers.json`。

## 建议关键词池（可每日固定订阅 + 二次过滤）

强相关（电商内容生态&达人治理）：e-commerce / shopping、shopping assistant / conversational commerce、product retrieval / search / ranking / rerank、vector database / ANN / embedding、query understanding / rewrite、content understanding / multimodal / caption、dedup / near-duplicate / similarity / clustering、violation detection / moderation / policy compliance、governance / integrity、data attribution / data selection、data cleaning / labeling。

弱相关但值得关注（大模型方向）：LLM/VLM/MLLM、agent / tool-use / memory、RAG、distillation、efficient post-training、robustness / evaluation。

## 推荐补充的 paper sources（便于每日抓新）

- 论文聚合：HuggingFace Daily Papers、Papers with Code（Trending / Latest）、arXiv-sanity。
- arXiv 入口：cs.IR/cs.CL/cs.CV/cs.LG/cs.AI 的 daily listing / RSS（再做关键词二次过滤）。
- 会议/评审：OpenReview（ICLR/NeurIPS/ICML/ACL 等）按 recent submissions / decisions；会议 accepted list（CVPR/ICCV/ECCV、KDD/SIGIR、SIGMOD/VLDB）。
- 学术检索：Semantic Scholar（topic alerts）、Google Scholar Alerts、OpenAlex（concept + date filter）、DBLP（会议/期刊增量）。
- 工业界产出：Google Research/DeepMind、Meta AI、Microsoft Research、Amazon Science、NVIDIA Research、Qwen、Hunyuan 等官方 blog/RSS。
- 社交媒体：X（research lab 账号 + trending papers）、机器之心/量子位/新智元等公众号（用于补充“热度很高”的弱相关论文线索）。

## 评分机制（100 分）

- 方法创新性：30
- 实验指标：15
- 实验质量：15
- 方法效率：10
- 方法泛化性：5
- 论文相关性：25

## 80+ 高质量论文复现

- OneRetrieval（2606.13533）：原文给出的 GitHub 链接仓库仅含 README（无实现），因此沿用本仓库已实现的 toy pipeline，并同步到 `2026-06-13/OneRetrieval`。
- Influcoder（2606.13668）：在 `2026-06-13/Influcoder` 提供可运行 toy pipeline（数据/模型/训练/测试），复现“LoRA 梯度 influence 排名 → CountSketch 压缩 → distill 到 encoder embedding”的关键流程。

> 说明：复现以“关键机制可跑通 + 最小闭环”为目标，使用合成/简化数据验证核心思想；与论文的真实数据、训练规模与线上系统仍有差距。
