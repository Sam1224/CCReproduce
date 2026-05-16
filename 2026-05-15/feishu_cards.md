# Feishu Cards — 2026-05-15 Daily AI Paper Inspection

> 以下为评分 ≥ 40 的论文飞书卡片（共 6 篇）。格式严格按照要求输出。

---

📄 标题：CQ-SID + EG-GRPO: Efficient Generative Retrieval for E-commerce Search with Semantic Cluster IDs and Expert-Guided RL
👥 作者：（待论文公开详细信息）
🔗 链接：https://arxiv.org/abs/2605.14434
📝 方法概述：针对电商搜索中双塔+ANN 架构对长尾查询泛化不足的问题，提出 CQ-SID（类目感知量化语义ID）和 EG-GRPO（专家引导群体相对策略优化）两个核心技术。CQ-SID 通过 3 层残差 VQ-VAE 引入类目感知一级码本，使同类目商品聚类到相同前缀 token 从而压缩搜索空间；EG-GRPO 在 GRPO 训练中注入真实 SID 序列作为专家锚点，稳定稀疏奖励下的策略学习。整体采用 4 阶段递进训练（码本学习→SFT→GRPO→在线迭代）。
💡 创新性分析：1）CQ-SID 将十亿级商品目录的生成式检索搜索空间压缩 ~3×，为行业落地奠定基础；2）EG-GRPO expert anchor 机制有效解决长序列 SID 奖励稀疏的信用分配问题，是对标准 GRPO 的重要实用改进；3）端到端生成式召回框架可内化多维相关性约束，无需离线 ANN 索引。评分 85/100（STRONG），代码已复现。
📊 关键指标：内部电商数据集 Recall@10 相对提升 +8.2%（vs 双塔+ANN），长尾查询 Recall@10 相对提升 +15.4%，在线 A/B GMV +1.3%（相对提升）。

---

📄 标题：From Scenes to Elements: Multi-Granularity Evidence Retrieval for Verifiable Multimodal RAG (GranuRAG)
👥 作者：Guanhua Chen, Chuyue Huang, Yutong Yao, Shudong Liu, Xueqing Song, Lidia S. Chao, Derek F. Wong
🔗 链接：https://arxiv.org/abs/2605.15019
📝 方法概述：现有多模态 RAG 系统在场景（整图）级别检索证据，导致细粒度查询失配且失败不可解释。本文提出 GranuRAG 框架，将视觉元素（visual elements）作为一等检索单元，包含三阶段：VLM 驱动的元素级检测与分类、基于 CLIP 的元素 embedding 相似度检索、以及多粒度证据融合生成。同时发布 GranuVistaVQA 基准（真实地标多粒度元素标注，含跨视角局部观察挑战）。通过显式元素 grounding 实现可验证的证据链。
💡 创新性分析：将检索粒度从场景下推至元素级是关键创新，避免 scene-query 粒度失配；可验证证据链支持错误诊断，超越黑盒 attention 机制；依赖现成 VLM 检测器和 CLIP，部署门槛低；GranuVistaVQA 填补元素级多视角 VQA 基准空白。可直接迁移至电商商品属性 QA 和内容核查场景。评分 72/100（STRONG）。
📊 关键指标：GranuVistaVQA 基准，准确率相对 6 个强基线提升 +29.2%（包括 GPT-4V direct 和 scene-level RAG）。

---

📄 标题：To See is Not to Learn: Protecting Multimodal Data from Unauthorized Fine-Tuning of Large Vision-Language Model (MMGuard)
👥 作者：（待论文公开详细信息）
🔗 链接：https://arxiv.org/abs/2605.14291
📝 方法概述：LVLM 普及使未授权图文数据微调（数据窃用）问题凸显。MMGuard 对多模态数据添加专用防护扰动，图像视觉上保持正常，但一旦被未授权微调则破坏视觉-语义对齐，使微调后模型失效。核心目标为最大化跨模态对齐破坏（KL 散度），同时保持原始感知质量（L-inf 约束）；支持白盒和黑盒场景。
💡 创新性分析：多模态数据保护是数字内容治理新兴方向；属于防御性内容安全，与 AIGC 检测互补形成"防+检"双保护；直接价值：为达人原创图文内容和电商商品主图提供版权技术防护，遏制 AI 商业窃用。评分 64/100（STRONG）。
📊 关键指标：未授权微调后 VQA 准确率相对下降 >40%（MMGuard 保护 vs 无保护），保护扰动 LPIPS 感知得分 ≤ 0.05（视觉不可察觉）。

---

📄 标题：Bad Seeing or Bad Thinking? Rewarding Perception for Vision-Language Reasoning
👥 作者：Haozhe Wang, Qixin Xu, Changpeng Wang, Taofeng Xue, Chong Peng, Wenhu Chen, Fangzhen Lin
🔗 链接：https://arxiv.org/abs/2605.14054
📝 方法概述：VLM 推理失败时，感知错误（"看错了"）和逻辑错误（"想错了"）混淆，难以定向优化。本文在 RLVR 框架中引入感知保真度奖励（Perception Fidelity Reward），对视觉 grounding 阶段单独打分，与最终答案奖励解耦，实现感知-推理协同 RL 优化。通过两步评价器分别计算感知分（视觉定位是否正确）和推理分，课程学习逐步提升感知难度。
💡 创新性分析：感知/推理双奖励解耦是对 RLVR 框架的重要补充（ICML 2026 Spotlight 认可）；对电商商品属性识别和内容审核 VLM 有直接改进价值，因感知错误是主要失败模式。评分 65/100（WEAK）。
📊 关键指标：MMStar +4.1% abs，MathVista +3.8% abs，VQAv2感知困难分割 +6.2% abs（vs Base RLVR）。

---

📄 标题：Towards On-Policy Data Evolution for Visual-Native Multimodal Deep Search Agents (ODE)
👥 作者：（待论文公开详细信息）
🔗 链接：https://arxiv.org/abs/2605.10832
📝 方法概述：多模态搜索 Agent 面临两个瓶颈：工具返回的中间图像被丢弃无法复用，以及静态训练数据与策略能力持续漂移。ODE 提出图像库引用协议（Image Bank Reference Protocol），将每个工具返回的图像注册为全局可寻址引用；同时设计在线策略数据演化闭环，用当前策略的 rollout 反馈（成功/失败轨迹）自动精炼训练数据，保持数据分布与策略能力同步。
💡 创新性分析：Image Bank 协议实现真正"视觉原生"多跳推理；ODE 闭环演化是对 self-play 的实用工程化，可泛化到电商多模态搜索场景。评分 62/100（WEAK）。
📊 关键指标：8 个多模态深度搜索 benchmark 验证，复杂多跳任务相对静态合成数据训练有显著提升。

---

📄 标题：HyperEyes: Dual-Grained Efficiency-Aware Reinforcement Learning for Parallel Multimodal Search Agents
👥 作者：DeepExperience team
🔗 链接：https://arxiv.org/abs/2605.07177
📝 方法概述：现有多模态搜索 Agent 顺序调用工具，多独立子检索任务导致轮次爆炸。HyperEyes 将视觉 grounding+检索融合为单一并行原子动作；TRACE 宏观奖励函数基于参考工具调用数单调收紧机制压缩无效调用；On-Policy Distillation 对失败 rollout 注入 teacher token 级纠错信号；首发 IMEB 效率感知 benchmark（300 实例，同时评估准确率和推理成本）。
💡 创新性分析：并行原子动作 + 效率 first-class 训练目标是对搜索 Agent RL 范式的实质性推进；IMEB 重新定义多模态搜索评测标准；代码开源（GitHub: DeepExperience/HyperEyes）。评分 60/100（WEAK）。
📊 关键指标：HyperEyes-30B 在 6 个搜索 benchmark 平均准确率 +9.9%（相对最强开源 Agent），工具调用轮次减少 5.3×。
