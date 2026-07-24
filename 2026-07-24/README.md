# 2026-07-24 电商内容生态 & 达人治理 Paper 巡检

本日巡检时间窗口按 **GMT+8** 计算，为 **2026-07-24 00:00:00 ~ 2026-07-24 23:59:59**。

## 1. 关键词补充与筛选口径

在用户给定关键词基础上，本轮额外补充了以下更贴近业务落地的检索词：

- 生成式推荐 / generative recommendation / hierarchical semantic ID
- 广告推荐 / sparse feature ranking / creator exposure / controllable recommendation
- 数据准备 / data construction / data quality evaluation / downstream utility
- 实时审核 / response guard / VLM moderation / streaming moderation
- 多模态内容理解 / multimodal verbalization / content-based recommendation

过滤规则：

- 优先保留与电商内容理解、推荐排序、广告特征治理、数据打标/清洗/质检、审核治理强相关的论文；
- 允许纳入少量与 LLM/VLM/MLLM 方法演进强相关、但对电商内容生态为弱相关的高热度论文；
- 显式跳过安全 / 后门 / jailbreak / prompt attack / 红队类论文。

## 2. 今日 source coverage

本轮优先覆盖了适合“今日新论文”场景的高时效来源：

- arXiv new submissions：`cs.IR`、`cs.CL`、`cs.CV`、`cs.AI`、`cs.LG`
- 论文项目页 / GitHub 仓库 / Hugging Face dataset 页面（用于核查代码与数据是否真实存在）
- 推荐、审核、数据准备等与电商治理强相关的工业/学术交叉方向论文

建议后续自动化继续补充的高时效 source pool：

- Paper 类：arXiv new、Hugging Face Daily Papers、Semantic Scholar、OpenAlex、DBLP、Papers with Code Trending、OpenReview 最新 revision/submission
- 企业 / 实验室：Google / DeepMind blog、Meta AI blog、OpenAI、Qwen Research、DeepSeek、腾讯混元、小红书/美团/快手技术博客
- 社交传播：X、机器之心、量子位、新智元、CVer、知乎 AI 热帖
- 会议 / 期刊：ICML / NeurIPS / ICLR、CVPR / ICCV / ECCV / TPAMI、ACL / EMNLP / NAACL、KDD / SIGIR、SIGMOD / VLDB 的 accepted paper 公告页

## 3. 今日入选论文（评分 >= 40）

共入选 7 篇：

1. `2607.21028` BARGE — **85**
2. `2607.21519` DLMRec — **83**
3. `2607.20465` DataPrep-Bench — **81**
4. `2607.20938` CCBR — **79**
5. `2607.20863` PRL — **78**
6. `2607.21401` ResponseGuard — **77**
7. `2607.20873` LO-FAR — **74**

详细中英双语摘要、标签、链接与分项评分见 [papers.json](./papers.json)。

## 4. 高分论文代码处理（评分 >= 80）

- `2607.21028`：未确认公开代码，已补充 toy PyTorch 复现，见 [BARGE](./BARGE)。
- `2607.21519`：已核查公开仓库，包含 `LightGCN`、量化模块、训练代码等真实实现，**不重复复现**。
- `2607.20465`：已核查公开仓库，包含 benchmark、data-selection、evaluation 等真实实现，**不重复复现**。

额外核查说明：

- `2607.21401` ResponseGuard 虽给出 GitHub 链接，但仓库当前主要为 README 与 docs 资源，未确认完整训练/推理代码；由于本轮评分未达到 80，先不触发复现。
- `2607.20938` CCBR 当前公开的是项目页，未在论文与项目页中直接确认完整训练仓库，因此暂列重点跟踪。

## 5. Web 展示

沿用现有 `paper_webapp`：

- 中英文切换
- 日期选择
- 分数滑动条过滤
- 论文标签 / 日期 / 作者 / 机构展示
- 论文链接与复现链接展示
- methodology figure 显示/隐藏
- 亮眼实验图与文字显示/隐藏
- SQLite 静态数据库按天存档

本轮补充今日 `papers.json`、BARGE 复现目录、7 篇论文的 methodology / experiment SVG 资源，并重建数据库与部署。