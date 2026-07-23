# 2026-07-23 电商内容生态 & 达人治理 Paper 巡检

本日巡检时间窗口按 **GMT+8** 计算，为 **2026-07-23 00:00:00 ~ 2026-07-23 23:59:59**。

## 1. 关键词补充与筛选口径

在用户给定关键词基础上，本轮额外补充了以下更贴近业务落地的检索词：

- 排序 / 精排 / reranking / ranking benchmark
- 推荐 / generative recommender / slate generation / creator exposure
- 证据检索 / setwise retrieval / evidence grounding / multimodal search
- AI 生成内容检测 / AIGC video detection / livestream moderation
- 多目标优化 / fairness / diversity / cold-start exposure
- 事实冲突 / 冗余控制 / 真实性评估 / rubric-based retrieval

过滤规则：

- 保留与电商内容理解、达人治理、推荐/检索基础设施、多模态证据治理、AIGC 检测相关的论文；
- 允许纳入少量与 Agent / RAG / MLLM 强相关、但对内容生态只有弱相关的高热度论文；
- **显式跳过安全 / 后门 / prompt 攻击 / 红队类论文**。

## 2. 今日 source coverage

本轮优先覆盖了适合“今日新论文”场景的高时效来源：

- arXiv new submissions：`cs.IR`、`cs.CL`、`cs.AI`、`cs.CV`、`cs.LG`
- arXiv / GitHub 项目页 / HuggingFace dataset 页面（用于补代码和数据链接）
- 论文项目页（如 Rubric4Setwise）

补充建议的日常高时效来源池如下，便于后续自动化扩展：

- Paper 类：arXiv new、OpenReview 最新 submission/revision、Semantic Scholar、OpenAlex、DBLP、Papers with Code Trending
- 企业 / 实验室：Google Research / DeepMind blog、Meta AI blog、OpenAI、Qwen Research、DeepSeek、腾讯混元、小红书 / 美团 / 快手技术博客
- 社交传播：X、机器之心、量子位、新智元、CVer、知乎 AI 热帖
- 会议 / 期刊：ICML / NeurIPS / ICLR、CVPR / ICCV / ECCV / TPAMI、ACL / EMNLP / NAACL、KDD / SIGIR、SIGMOD / VLDB 的 accepted paper 公告页

备注：本轮 arXiv API 与 Google Scholar 抓取存在限流 / 抓取失败，因此主依赖 arXiv new listing 页面完成当天筛选。

## 3. 今日入选论文（评分 >= 40）

共入选 6 篇：

1. `2607.19747` Rubric4Setwise — **94**
2. `2607.19987` UniRank — **93**
3. `2607.19739` PRTA — **87**
4. `2607.19476` Detect Early, Escalate Rarely — **84**
5. `2607.19357` SPD Decoding for Generative RS — **82**
6. `2607.19793` Silent Failures in Multimodal Agentic Search — **76**

详细中英双语摘要、分项评分、链接与标签均已写入 [papers.json](./papers.json)。

## 4. 高分论文代码处理（评分 >= 80）

- `2607.19747`：已核查原项目，仓库包含 `run_pipeline.py`、`judge_scoring/`、`rankify/` 等实现，**不重复复现**。
- `2607.19987`：已核查原项目，仓库包含 `model_zoo/`、`benchmark/`、`unirank/` 等实现，**不重复复现**。
- `2607.19739`：已核查原项目，仓库包含 `main.py`、`memory_bank.py`、`SimpleX/`、`SASRec/`、`LightGCN-PyTorch/` 等实现，**不重复复现**。
- `2607.19476`：已核查原项目，仓库包含 `streamdet/`、`analysis/`、`features/`、`tests/` 等实现，**不重复复现**。
- `2607.19357`：未确认公开实现，已补充 toy PyTorch 复现，见 [SPD_Decoding](./SPD_Decoding)。

## 5. Web 展示

现有 `paper_webapp` 已支持：

- 中英文切换
- 日期选择
- 分数滑动条过滤
- 论文标签 / 日期 / 作者 / 机构展示
- 论文链接与复现链接展示
- methodology figure 显示/隐藏
- 亮眼实验文本显示/隐藏
- SQLite 静态数据库按天存档

本轮只需补充今日 `papers.json`、methodology figures，并重建数据库与部署即可。
