# Feishu Cards — 2026-06-16 Daily AI Paper Inspection

> Papers with score ≥ 40. Format: one card block per paper.

---

📄 标题：Dynamic Content Moderation in Livestreams: Combining Supervised Classification with MLLM-Boosted Similarity Matching
👥 作者：Wei Chee Yew et al. (ByteDance / TikTok)
🔗 链接：https://arxiv.org/abs/2512.03553
📝 方法概述：双路径直播内容审核系统——监督分类管道检测已知违规，相似度匹配管道识别新型违规；两条管道均通过MLLM知识蒸馏提升精度，同时保持轻量推理；处理文本+音频+视觉三模态输入；KDD 2026 工业论文。
💡 创新性分析：首次将MLLM蒸馏与双路径多模态审核系统结合并在字节跳动直播电商平台大规模验证；相似度管道有效应对传统分类器无法覆盖的新型违规，形成互补覆盖机制。
📊 关键指标：TikTok生产环境 | 分类管道 Recall@80%P=67%，相似度管道 Recall@80%P=76% | A/B测试用户观看违规直播减少6~8%（分数：88/100）

---

📄 标题：UNIVID: Unified Vision-Language Model for Video Moderation
👥 作者：Kejuan Yang*, Yizhuo Zhang* et al. (ByteDance)
🔗 链接：https://arxiv.org/abs/2606.05748
📝 方法概述：字节跳动提出的全球规模视频内容审核统一VLM。核心创新：生成策略感知字幕（Policy-Aware Caption）作为可解释中间表示，替代黑盒分类器。三阶段级联架构：①风险过滤器（多模态融合）②审核执行器（UNIVID-Lite快速决策+UNIVID-RAG历史案例召回）③趋势治理（嵌入聚类发现新兴违规）。
💡 创新性分析：将多个分类任务统一为策略感知文本生成，字幕同时服务于决策、人工审核和趋势检测；UNIVID-RAG解决传统规则无法覆盖的新型违规召回；Trend Governance自动发现新兴违规趋势，无需人工标注新类别。
📊 关键指标：ByteDance内部视频审核数据集 | 多任务统一审核指标优于专用分类器 | 可解释性显著提升（分数：83/100）

---

📄 标题：LLM-Based User Personas for Recommendations at Scale
👥 作者：Haoting Wang, Haokai Lu, Zheyun Feng et al. (Google)
🔗 链接：https://arxiv.org/abs/2606.12198
📝 方法概述：Google提出十亿级视频推荐平台上的在线LLM用户兴趣画像生成框架。通过语义聚类视频表示压缩用户历史，在线生成自然语言画像，知识蒸馏到小模型+异步推理实现低延迟在线serving；兼顾兴趣摘要与新话题探索。
💡 创新性分析：首次在十亿级在线推荐系统中实现LLM推理画像生成；语义聚类+KD+异步三重效率机制解决大规模在线推理瓶颈；LLM画像同时解决探索-利用均衡问题。
📊 关键指标：Google视频推荐平台 | 在线A/B CTR和用户时长显著提升 | 推理延迟满足在线serving要求（分数：77/100）

---

📄 标题：QueryAgent-R1: Bridging Query Generation and Product Retrieval for E-Commerce Query Recommendation
👥 作者：Dike Sun, Zheng Zou, Jingtong Zang et al. (Alibaba International Digital Commercial Group)
🔗 链接：https://arxiv.org/abs/2606.05671
📝 方法概述：阿里巴巴国际提出QueryAgent-R1，记忆增强RL智能体用于电商查询推荐。链式检索优化：生成查询→真实库存检索→验证/优化查询，通过一致性奖励联合优化查询CTR和商品CVR。
💡 创新性分析：首次将真实库存检索结果作为RL奖励信号融入查询生成过程，实现CTR-CVR端到端对齐；一致性奖励设计解决查询推荐与下游商品转化脱节的行业痛点。
📊 关键指标：阿里巴巴国际电商平台在线A/B测试 | CTR和CVR双提升 | 端到端商业指标显著正向（分数：77/100）

---

📄 标题：Customer-Agent: Overcoming Context Limitations in Ultra-Long Shopping Trajectories via Tool-Augmented Agents and RLVR
👥 作者：Hongye Liu, Rongmei Lin, Anurag Kashyap et al. (Amazon, Duke University)
🔗 链接：https://arxiv.org/abs/2606.07995
📝 方法概述：针对用户多年购物历史远超LLM上下文窗口的问题，Customer-Agent将轨迹外化存储为文件，训练LLM智能体通过代码解释器工具调用自主检索解析；构建ShopTrajQA基准，RLVR训练保证推理准确性。
💡 创新性分析：工具增强轨迹外化突破LLM上下文限制；RLVR训练在购物轨迹推理任务上的首次应用；ShopTrajQA为超长购物历史理解提供首个标准评测基准。
📊 关键指标：ShopTrajQA基准 | 超长轨迹推理准确率优于标准LLM | 工具调用方式优于RAG基线（分数：71/100）

---

📄 标题：LLM Judges Have Dark Current: A Psychometric Datasheet for LLM-as-a-Judge Evaluation
👥 作者：Hiroyasu Usami, Keisuke Hara, Ayato Tsuboi, Naohiko Matsuda
🔗 链接：https://arxiv.org/abs/2606.15610
📝 方法概述：从心理测量学视角系统审计LLM评判器的五类偏差：暗电流（无效输入的虚假判断）、跨敏感性、位置偏好、目标敏感性、判断准则偏移；提出Judge Datasheet协议作为标准化报告工具。
💡 创新性分析：将心理测量学引入LLM评判器质量评估，方法论创新；暗电流概念形象揭示评判器在无有效信号时的本底偏差；为电商内容审核、数据标注流水线中的LLM评判器选型提供实用工具。
📊 关键指标：三个开源评判器（Llama-3.1-8B, Qwen2.5-14B, Qwen2.5-32B）| Llama-3.1-8B暗电流高且存在位置冲突 | Qwen2.5-14B暗电流低但有稳定/位置歧视混合（分数：66/100）

---

📄 标题：MiniMax Sparse Attention
👥 作者：Xunhao Lai, Weiqi Xu, Yufeng Yang et al. (MiniMax)
🔗 链接：https://arxiv.org/abs/2606.13392
📝 方法概述：MiniMax提出MSA（分块稀疏注意力），轻量级Index Branch对KV块评分并为每个GQA组独立选择Top-k块；Main Branch执行精确块稀疏注意力。支持1M token上下文，计算量降至全注意力1/20。已集成到MiniMax M3模型。
💡 创新性分析：组特定稀疏检索比全局稀疏更精确；工程设计简洁可扩展；1M上下文为长文档/长视频内容分析提供基础设施支撑。
📊 关键指标：1M tokens上下文 | 计算量约1/20全注意力 | Prefill加速9.7× | Decode加速15.6×（分数：65/100）

---

📄 标题：CORA: Analyzing and Bridging Thinking-Answer Gap in Multimodal RLVR via Consistency-Oriented Reasoning Alignment
👥 作者：(待确认)
🔗 链接：https://arxiv.org/abs/2606.14691
📝 方法概述：针对多模态RLVR训练中思维链与最终答案语义矛盾问题，提出即插即用一致性奖励模型（CRM），量化thinking-answer语义一致性并作为额外RL奖励信号，无需修改基础训练框架。
💡 创新性分析：首次明确分析多模态RLVR中thinking-answer gap问题；CRM即插即用设计降低集成成本；对多模态内容理解、产品审核等推理任务有参考价值。
📊 关键指标：多个多模态推理基准 | 一致性奖励改善推理准确率 | 思维-答案语义一致性显著提升（分数：61/100）

---

📄 标题：Who Drifted: the System or the Judge? Anytime-Valid Attribution in LLM Evaluation Pipelines
👥 作者：Yitao Li
🔗 链接：https://arxiv.org/abs/2606.15474
📝 方法概述：解决LLM评判器持续监控中评分漂移归因歧义问题。通过固定人工标注锚集+双重e-process（赌注过程）统计方法，守卫窗口规则实时返回漂移归因：{无漂移、系统漂移、评判器漂移}，提供任意时刻有效性理论保证。
💡 创新性分析：首次用统计严格方法解决生产环境中LLM评判器漂移归因歧义问题；锚集方案轻量低成本；对内容审核质量监控流水线有直接应用价值。
📊 关键指标：理论证明任意时刻有效性 | 模拟实验验证归因准确性 | 一方式区分系统vs评判器漂移（分数：58/100）
