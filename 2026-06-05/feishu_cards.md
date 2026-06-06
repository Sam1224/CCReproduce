# Feishu Cards — 2026-06-05

Papers with score ≥ 40 (all 6 selected papers):

---

📄 标题：UNIVID: Unified Vision-Language Model for Video Moderation
👥 作者：Kejuan Yang, Yizhuo Zhang, Mingyuan Du, Yue Zhang, Dixin Zheng, Kaili Zhao, Yang Xiao, Hanzhong Liang, Kenan Xiao（ByteDance）
🔗 链接：https://arxiv.org/abs/2606.05748
📝 方法概述：UNIVID 是 ByteDance 提出的统一视频内容审核视觉-语言模型。它以"政策感知字幕"作为可解释中间表示，单一模型覆盖全部审核策略，用字幕内容替代不透明分类器的决策逻辑，既支持人工审核员可读验证，又实现多政策复用。训练采用专家精标与合成数据混合配方，规避了通用 VLM 的安全护栏误拒问题。
💡 创新性分析：将生成式 VLM 用于内容审核是方法论范式转变；政策感知字幕兼具可解释性与多任务复用能力；用一个模型替代 1000+ 专用分类器，大幅降低工程与计算成本；显式解决安全护栏拒识问题是工业落地的关键创新。
📊 关键指标：ByteDance 生产平台；违规漏报率下降 42.7%（相对）；误判率下降 37.0%（相对）；替代 1000+ 个策略专用模型 | 综合评分：87/100

---

📄 标题：ANCHOR: Agentic Noise Creation Framework for Human Simulation and Denoising Recommendation
👥 作者：（论文作者待完整索引，arXiv:2606.05621）
🔗 链接：https://arxiv.org/abs/2606.05621
📝 方法概述：ANCHOR 通过"创造-识别循环"解决隐式反馈去噪难题：基于 LLM 的智能体模拟器生成五类噪声标注（误点击、猎奇、标题偏差、热门偏差、位置偏差）的合成交互数据，轻量噪声识别器在此基础上训练，再应用于真实日志去噪，最终用干净数据训练推荐模型，形成闭环。推荐器在环的边界精化使合成噪声分布更贴近真实系统。
💡 创新性分析：无需人工噪声标注，用 LLM 智能体"先造噪声再识别"，解决了行业痛点；五类精细化噪声分类直接对应电商场景常见污染源；方法与推荐模型解耦，可即插即用。
📊 关键指标：Yelp Recall@20 +3.1%；Amazon-Books NDCG@20 +2.6%；MovieLens Recall@20 +4.3%（均相对最强基线）| 综合评分：84/100

---

📄 标题：QueryAgent-R1: Bridging Query Generation and Product Retrieval for E-Commerce Query Recommendation
👥 作者：Dike Sun, Zheng Zou, Jingtong Zang, Qi Sun, Huaipeng Zhao, Tao Luo, Xiaoyi Zeng（Alibaba International Digital Commercial Group）
🔗 链接：https://arxiv.org/abs/2606.05671
📝 方法概述：QueryAgent-R1 是阿里国际电商查询推荐智能体框架，通过链式检索优化（chain-of-retrieval optimization）弥合查询生成与商品检索的对齐鸿沟：生成候选查询词 → 执行真实库存检索 → 利用一致性奖励（同时优化查询相关性与下游商品参与度）进行强化学习微调。记忆抽象模块从用户长期行为提取兴趣图谱，支持高效用户画像。
💡 创新性分析：首次将真实商品库存检索嵌入查询推荐 RL 训练循环；一致性奖励同时对齐点击率与转化率，跳出"优化点击不优化转化"的陷阱；在日活数千万平台完成在线验证。
📊 关键指标：在线 A/B 测试；Query CTR +2.9%；Guided CVR +3.1% | 综合评分：79/100

---

📄 标题：OneReason Technical Report
👥 作者：OneRec Team（84位作者，快手）
🔗 链接：https://arxiv.org/abs/2606.06260
📝 方法概述：OneReason 是快手 OneRec 生成式推荐模型家族的最新工作，专注于解决生成式推荐中链式思维（CoT）推理无效的根本原因。研究识别两大鸿沟：感知鸿沟（物品 token 缺乏语言语义锚定）与认知鸿沟（模型缺少推荐专用推理范式），并通过联合学习物品感知与推荐认知，首次使"思考模式"在快手真实基准上稳定优于"非思考模式"。
💡 创新性分析：首次系统诊断生成式推荐中 CoT 失效原因；双鸿沟框架为推荐推理研究提供理论基础；在快手短视频、直播、电商三大场景验证。
📊 关键指标：快手内部基准；思考模式 vs 非思考模式在短视频/直播/电商多场景均有显著提升 | 综合评分：69/100

---

📄 标题：Organizational Control Layer: Governance Infrastructure at the Execution Boundary of LLM Agent Systems
👥 作者：McGill / Purdue / UNSW / UCLA / NYU / Stevens Institute 联合团队
🔗 链接：https://arxiv.org/abs/2606.04306
📝 方法概述：OCL 是部署在 LLM 智能体"生成层"与"执行层"之间的模型无关治理基础设施，拦截生成的动作并应用声明式策略规则（自然语言或结构化逻辑），在操作提交环境前完成放行/修改/人工升级的决策。在多个前沿 LLM 后端的对抗性买卖谈判场景中评估。
💡 创新性分析：将执行边界问题抽象为独立的基础设施层，实现与模型架构解耦；策略规则可用自然语言表达，易于运营团队配置和维护。
📊 关键指标：对抗性谈判环境；不安全执行率：88%→~0%；有效成功率：12%→96% | 综合评分：61/100

---

📄 标题：DetectZoo: A Unified Toolkit for AI-Generated Content Detection Across Text, Audio, and Image Modalities
👥 作者：Sajad Ebrahimi 等（多机构学术团队）
🔗 链接：https://arxiv.org/abs/2606.04205
📝 方法概述：DetectZoo 是首个跨文本、音频、图像三种模态的统一 AIGC 检测工具箱，提供 61 个检测器实现、22 个基准数据集加载器和标准化评估流水线。每个检测器自包含，自动下载预训练权重，复现已发布基线结果，大幅降低 AIGC 检测研究的入门和复现门槛。
💡 创新性分析：标准化整个 AIGC 检测实验流程，解决领域碎片化问题；覆盖三种模态，支持跨模态公平对比；开源 Python 包（PyPI: detectzoo）降低工程门槛。
📊 关键指标：61 个检测器 + 22 个数据集 + 文本/图像/音频三模态 + 复现率 100%（声明）| 综合评分：59/100
