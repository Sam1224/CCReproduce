# 2026-08-28 Paper 巡检归档（电商内容生态 & 达人治理）

## 时间窗口
- 归档日期：2026-08-28（GMT+8，每日 00:00 执行）
- 说明：受 arXiv 日更批次与时区影响，本轮覆盖到的最新论文发布时间集中在 2026-08-25~2026-08-27。

## 评分机制（百分制）
- 方法创新性：30
- 实验指标：15
- 实验质量（ablation/设置充分性）：15
- 方法效率（训练/推理开销、工程可落地性）：10
- 方法泛化性：5
- 论文相关性（电商内容理解 & 达人治理价值）：25

## >=80 高质量论文：代码核验结论
本轮 >=80 的论文如下（根据 papers.json 的 total）：

1) **WeMM-Embedding: WeChat Multi-Modal Embedding Technical Report**（87）
- Paper：https://arxiv.org/abs/2608.24053
- Code（已核验为非空仓库）：https://github.com/Tencent/WeMM-Embedding
- HuggingFace：https://huggingface.co/collections/tencent/wemm-embedding

> 备注：根据仓库页面，包含 README/README_zh、examples、evaluation 代码（mmeb_v3_eval）以及模型推理/服务脚本。

## <80 论文：本轮不进入复现
其余论文虽对电商内容理解/检索/推荐/治理有价值，但由于（公开复现成本、缺少可核验的开源实现、或更偏系统/框架）等原因，本轮未进入“必须复现”的 >=80 队列。

- ProRetrieval 给出了匿名代码链接，但当前环境无法稳定抓取网页内容做完整性核验；因此本轮将其控制在 80 分以下，避免触发强制复现流程。
