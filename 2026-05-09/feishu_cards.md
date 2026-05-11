# Feishu Cards — 2026-05-09 Daily AI Paper Inspection
# 飞书卡片 — 2026-05-09 日报 (分数 ≥ 40 的论文)

---

📄 标题：GLiGuard: Schema-Conditioned Classification for LLM Safeguard
👥 作者：GLiNER2 团队衍生研究团队
🔗 链接：https://arxiv.org/abs/2605.07982
📝 方法概述：提出 GLiGuard，将 LLM 护栏重定义为 schema-conditioned 多任务分类问题：给定输入 schema（指定审核任务），基于 0.3B 参数双向编码器在**单次非自回归前向传播**中同时完成 prompt 安全、response 安全、拒绝检测、14 类细粒度有害类别及 11 种越狱策略检测，23–90× 小于竞争对手（7B–27B），推理吞吐量提升 16×，延迟降低 17×。
💡 创新性分析：首次将护栏问题从"自回归生成"转为"schema 条件化多任务分类"，架构突破性显著；0.3B 参数实现与大模型护栏相当的 F1，极大降低工业部署成本。评分：83/100（STRONG）
📊 关键指标：F1 与 7B–27B 解码器持平（WildGuard/PromptBench）；Throughput 16×↑；Latency 17×↓；参数 0.3B（比 LlamaGuard-3-8B 小 ~27×）

---

📄 标题：Who Decides What Is Harmful? Content Moderation Policy Through A Multi-Agent Personalised Inference Framework
👥 作者：未完全披露
🔗 链接：https://arxiv.org/abs/2605.01416
📝 方法概述：提出基于 LLM 的多智能体个性化内容审核框架：领域专家 Agent 进行专项有害性判断，Manager Agent 汇总协调裁决，Ghost Profile Agent 根据用户历史行为构建隐性敏感度档案以动态调整审核阈值。将"谁来决定什么有害"的问题从统一平台层下移至个性化用户层，同时保持平台政策基线约束。
💡 创新性分析：Ghost Profile Agent 实现用户级个性化审核阈值，首次系统性将用户档案引入内容审核决策链；三层多 Agent 协同架构可扩展至新有害内容类别，无需重新训练专有模型。评分：75/100（STRONG）
📊 关键指标：相比非个性化基线准确率最高提升 **+32%**；多类有害内容分类 F1 全面优于统一策略基线

---

📄 标题：Algospeak, Hiding in the Open: The Trade-off Between Legible Meaning and Detection Avoidance
👥 作者：Jan Fillies, Ronald E. Robertson, Jeffrey Hancock 等
🔗 链接：https://arxiv.org/abs/2605.06619
📝 方法概述：系统形式化 Algospeak（算法语，达人用于规避内容审核的语言改写策略）的博弈动态。提出 MUM（Majority Understandable Modulation）概念，定义逃逸-理解 trade-off 的甜蜜点。构建 700 个标注样本（20 基础句 × 5 调制级别 × 7 规避策略），用 7 个主流 LLM 验证意义恢复能力与检测规避能力的系统性 trade-off。
💡 创新性分析：首次形式化 Algospeak 最优规避点（MUM）并提供可复现评测框架；揭示轻度调制即可大幅降低检测率，为平台防御提供理论依据和对抗数据集构建方法。评分：68/100（STRONG）
📊 关键指标：7 个 LLM 检测一致性较弱；轻度调制（Level 1-2）检测率大幅下降且语义保留 >90%；MUM 甜蜜点约在 Level 3

---

📄 标题：Lightweight Stylistic Consistency Profiling: Robust Detection of LLM-Generated Textual Content for Multimedia Moderation
👥 作者：Siyuan Li 等 9 位作者
🔗 链接：https://arxiv.org/abs/2605.05950
📝 方法概述：提出 LiSCP，利用"AI 生成文本风格一致性强于人类文本"这一本质特性，通过多模态引导改写变体构建文本风格一致性画像。结合离散风格特征（词法/句法）与连续语义信号（嵌入相似度），实现对改写攻击具有内在鲁棒性的轻量 AI 文本检测，专为多媒体内容审核场景设计。
💡 创新性分析：从一致性视角检测 AI 内容而非依赖统计启发（困惑度等），对改写攻击内在鲁棒；轻量化设计适合内容平台大规模部署，可用于电商评论/达人内容真实性检测。评分：63/100（STRONG）
📊 关键指标：改写攻击下 AUROC 显著优于 GPTZero/DetectGPT 等基线；跨多种 LLM 生成源保持泛化性

---

📄 标题：LatentRAG: Latent Reasoning and Retrieval for Efficient Agentic RAG
👥 作者：未完全披露
🔗 链接：https://arxiv.org/abs/2605.06285
📝 方法概述：将 Agentic RAG 的多步推理和检索全部迁移至连续隐变量空间：LLM 不再自回归生成中间推理步骤和子查询文本，而是直接产生隐状态，与密集检索模型在同一嵌入空间对齐，实现单次前向传播完成多步推理+检索。并行隐解码机制将隐状态映射回自然语言以保持可解释性，端到端联合优化 LLM 与检索模型。
💡 创新性分析：消除 Agentic RAG 最大的延迟瓶颈（自回归文本生成）；90% 延迟降低是工业级突破，使在线 RAG 应用（如电商智能问答）变得实用。评分：73/100（WEAK）
📊 关键指标：7 个多跳 QA 基准精度与显式 Agentic RAG 相当；推理延迟降低 **~90%**；大幅缩小与单步 RAG 的延迟差距

---

📄 标题：Mamoda2.5: Enhancing Unified Multimodal Model with DiT-MoE
👥 作者：ByteDance Seed 团队
🔗 链接：https://arxiv.org/abs/2605.02641 | GitHub: https://github.com/bytedance/mammothmoda
📝 方法概述：提出统一 AR-Diffusion 框架，以 DiT 为骨干引入细粒度 MoE（128专家/Top-8路由），形成 25B 总参数仅激活 3B 的稀疏模型，在单一模型中统一支持文生图、文生视频、图像/视频编辑、多模态理解。联合少步蒸馏+RL 将 30 步编辑模型压缩至 4 步，大幅加速推理。
💡 创新性分析：AR+Diffusion+DiT-MoE 三重融合，实现参数高效的多任务统一模型；4 步推理蒸馏（原 30 步）工程创新明显；字节跳动 Seed 团队出品，工业可信度高。评分：67/100（WEAK）
📊 关键指标：VBench 2.0 开源 Top 级；OpenVE-Bench 视频编辑质量与 Kling O1 持平；推理 4 步（原 30 步）；激活参数仅 3B/25B total

---

📄 标题：Intent-Driven Semantic ID Generation for Grounded Conversational News Recommendation
👥 作者：未完全披露
🔗 链接：https://arxiv.org/abs/2605.07613
📝 方法概述：针对对话式新闻推荐中"动态语料库+模糊意图"双重挑战，提出意图驱动 SID（语义ID）生成框架：从生产对话归纳 6 类意图类型，LLM 将意图映射为分层语义 ID 前缀，通过模糊匹配与实时新闻池对接（保证推荐完全接地），两阶段训练（SID 对齐 + GPT-4 CoT 蒸馏）强化意图理解。
💡 创新性分析：解耦意图层与检索层，通过分层 SID 前缀解决动态语料库可检索性问题；GPT-4 CoT 蒸馏将强推理能力迁移至轻量学生模型，适用于高频更新内容场景。评分：62/100（WEAK）
📊 关键指标：Recall@K 优于直接 ID 生成基线；6 类意图匹配准确率达 SOTA；接地率（有效推荐率）显著提升

---

📄 标题：From Passive Feeds to Guided Discovery: AI-Initiated Interaction for Vague Intent in Content Exploration (Red-Rec)
👥 作者：Yu Xie, Ying Qi
🔗 链接：https://arxiv.org/abs/2605.02902
📝 方法概述：提出 Red-Rec，一个 AI 主动发起交互的内容探索框架：AI 自动归纳用户 Feed 内容模式，主动推送可点击的探索方向选项，最多追问一次，然后渐进式将新类型内容融入原 Feed。专为解决"用户 Feed 单调但意图模糊、无法明确表达新需求"的信息茧房问题而设计。
💡 创新性分析：将推荐主动权从被动响应转移至 AI 主动发起，适配意图模糊用户；设计约束（追问≤1次、渐进融入）平衡探索体验与用户打扰成本；适用于小红书等内容/社交电商平台 Feed 优化。评分：57/100（WEAK）
📊 关键指标：用户研究显示相比用户主动对话推荐，Red-Rec 带来更广泛内容探索（多样性更高）且交互负担更低
