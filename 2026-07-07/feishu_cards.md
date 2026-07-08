# Feishu Cards — 2026-07-07

> 以下卡片适用于每日论文巡检飞书推送（score ≥ 40 的论文）。

---

📄 标题：Yuvion VL: A Multimodal Foundation Model for Adversarial Content and AI Safety
👥 作者：Ting Cao et al.（Yuvion AI）
🔗 链接：https://arxiv.org/abs/2606.25034
📝 方法概述：面向内容安全与 AI 安全的 8B/32B 多模态大模型，核心三件套：① 对抗感知数据飞轮（挖掘规避策略如小字/logo 变形/遮挡/AI 伪装，持续生成对抗训练数据）；② C2FT（Confuse-then-Contrast Fine-Tuning，在线挖掘模型混淆对，多图联合对比监督）；③ YVRE 三级评估框架（开源通用 → 安全 → 电商商业 benchmark，含 logo 识别、品牌仿冒检测、商品类目合规、价格合规四类治理场景）。
💡 创新性分析：相比 Llama-Guard 等静态安全数据集训练的模型，Yuvion VL 构建对抗飞轮闭环（失败案例 → 有针对性增强），C2FT 把混淆挖掘嵌入训练循环；YVRE 将电商治理指标纳入评估，直接覆盖平台合规与达人违规场景，是现有安全模型普遍缺失的维度。
📊 关键指标：YVRE 评估均分高出可比开源模型 +9.9 pts、高出更大闭源模型（GPT-5.4、Qwen3.5-Plus 等）+6.7 pts；Yuvion VL-8B 在多项安全任务上超越大部分 SOTA 基线（含参数更大的模型）。**总分 82/100**

---

📄 标题：MatchLM2Lite: A Scalable MLLM-to-Lite Framework for Reproduced Content Identification
👥 作者：Xiaotian Fan, Hiok Hian Ong, David Yuchen Wang, Zirui Zhu, Kanchan Sarkar, Kun Xu（TikTok）
🔗 链接：https://arxiv.org/abs/2606.14786
📝 方法概述：面向生产规模搬运内容识别（RCI）的两阶段框架：MatchLM（高容量 MLLM 教师）联合建模视频+音频+文本三模态，以视频对为输入输出细粒度重放分数；MatchLite（轻量学生）通过知识蒸馏继承教师能力，推理成本降低 35×。关键设计：把 MLLM 作为语义表示提取器而非生成式模型，直接用富语义 embedding 做判别分类，蒸馏目标更清晰。
💡 创新性分析：区别于像素哈希或音频指纹，MatchLM2Lite 用 MLLM 捕捉语义层面内容重用（覆盖剪辑、配音替换、字幕改写等规避手段）；将 MLLM 从生成器改造成紧凑语义提取器是关键技术贡献，35× 成本削减使其满足大规模内容平台实时审核需求。
📊 关键指标：MatchLM 相比前代生产模型 F1-score +8.57；蒸馏后 MatchLite 保留 +6.55 F1，计算成本降低 35×（推理速度大幅提升）。**总分 80/100**

---

📄 标题：Bridging Short Videos and Live Streams: Reasoning-Guided Multimodal LLMs for Cross-Domain Representation Learning
👥 作者：Le Zhang, Xiaolan Zhu, Yuchen Wang et al.（快手科技 Kuaishou Technology / 中国人民大学）
🔗 链接：https://arxiv.org/abs/2606.04448
📝 方法概述：RGCD-Rep 两阶段框架解决短视频→直播冷启问题。①推理感知蒸馏：冻结教师 MLLM 生成结构化跨域推理知识，蒸馏进轻量学生 MLLM；②可迁移性引导的跨域表示学习：item 表示分解为可迁移表示（跨域共通语义）+领域残差（直播特有信息），结合行为协作信号对齐。表示可离线预计算，低成本接入工业检索系统。
💡 创新性分析：已有跨域推荐多迁移用户行为，RGCD-Rep 改为迁移 item 语义表示并以 MLLM 推理为弱监督标签，降低标注成本；正交分解避免在稀疏直播行为上过拟合，泛化性更强；快手生产部署 A/B 实验验证。
📊 关键指标：相对最强 baseline 提升 +17.59% 和 +30.93%；快手直播推荐 A/B 实验多个核心业务指标显著提升。**总分 77/100**

---

📄 标题：SSRLive: Live Streaming Recommendation with Dynamic Semantic ID
👥 作者：Teng Shi, Zhaoheng Li, Yuanhang Qu, Yi Liu, Lixiang Lai, Yuning Jiang（阿里巴巴 Taobao & Tmall Group）
🔗 链接：https://arxiv.org/abs/2606.06970
📝 方法概述：统一生成-判别框架解决直播内容实时变化下的 item 表示漂移。编码器-解码器生成静态 SID（直播间持续属性）+动态 SID（实时内容状态），充分利用多模态内容；判别模块融合 SID+用户特征+主播-用户交互数据进行多任务预测，端到端联合训练。
💡 创新性分析：现有方法用固定 item 嵌入无法捕捉直播内容动态（节目切换、促销更新）；动态 SID 实时刻画内容状态，生成+判别统一框架将内容理解与协同过滤协同优化，全量上线服务数亿用户验证工程可行性。
📊 关键指标：线上 A/B 实验：观看时长 +3.38%、GMV +0.72%、新粉丝 +3.12%、互动量 +2.92%；已全量上线。**总分 74/100**

---

📄 标题：DSIRM: Learning Query-Bridged Discrete Semantic Identifiers for E-commerce Relevance Modeling
👥 作者：Zhenghao Liu et al.（阿里巴巴 Taobao & Tmall Group）
🔗 链接：https://arxiv.org/abs/2606.04374
📝 方法概述：双路提升离散语义 ID（SID）的电商搜索相关性表达：① Item 侧：query-bridged 对比量化（CBCQ）在残差量化中注入 query-item 交互监督，使 SID 分区形成相关感知语义划分；② Query 侧：生成式 LLM 从 query 文本直接预测 item SID，解决长尾 query 和意图歧义。层次前缀匹配特征与密集检索信号互补。
💡 创新性分析：现有 SID 方法依赖无监督量化，缺乏 query 相关感知；CBCQ 把 query-item 交互信号注入量化使码本形成"相关感知语义区间"；LLM 生成式补全 query 侧意图；两路互补提升相关性判别，已在阿里电商搜索系统部署。
📊 关键指标：层次前缀匹配特征与密集信号互补，带来显著相关性提升（具体数值见论文）；生产规模数据集验证，线上部署。**总分 73/100**

---

📄 标题：Yuvion LLM: An Adversarially-Aware Large Language Model for Content And AI Safety
👥 作者：Ting Cao et al.（Yuvion AI）
🔗 链接：https://arxiv.org/abs/2606.27632
📝 方法概述：Yuvion VL 的纯文本孪生模型，专注文本层面内容安全与 AI 安全，采用与 VL 版本相同的对抗感知数据飞轮和对比监督训练，在文本安全 benchmark 上达到 SOTA。适用于文本违规检测、LLM 越狱防护与内容合规判断。
💡 创新性分析：延续对抗感知训练范式在纯文本领域的验证；与 Yuvion VL 配合可覆盖文本+多模态内容安全完整链路，作为平台 API 安全网关的基础模型价值明确。
📊 关键指标：文本安全 benchmark SOTA（具体数值见论文）；与 Yuvion VL 配合覆盖完整内容安全栈。**总分 63/100**
