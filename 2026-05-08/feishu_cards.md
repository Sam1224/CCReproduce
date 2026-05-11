# 飞书卡片汇总 — 2026-05-08 (评分 ≥40，共 10 篇)

> 收件人: qianxiong.xu@bytedance.com
> 因当前环境未配置飞书 MCP，本文件为飞书卡片**等价 Markdown**，可直接粘贴到飞书富文本/卡片机器人。

---

📄 标题：UniVA: Unified Value Alignment for Generative Recommendation in Industrial Advertising
👥 作者：Xinxun Zhang, Yuling Xiong, Jiale Zhou 等 16 人 (Tencent, WeChat Channels)
🔗 链接：https://arxiv.org/abs/2605.05803
📝 方法概述：将工业广告生成式推荐拆为三步价值对齐：Commercial SID 把价值属性注入 token 表征；Generation-as-Ranking decoder 用 SFT + eCPM-aware RL 联合训练，把生成与排序合二为一；value-guided 个性化 beam search 配 trie 约束在线解码到合法 SID。
💡 创新性分析：从 token 层、解码层、serving 层三个维度同时做价值对齐，抹平离线-在线 gap。Commercial SID 作为值得借鉴的新组件。
📊 关键指标：Tencent WeChat Channels 离线 Hit Rate@100 +37.04%；在线 A/B GMV +1.5%。**得分 86，已复现 (code/UniVA/)**

---

📄 标题：CMTA: Leveraging Cross-Modal Temporal Artifacts for Generalizable AI-Generated Video Detection
👥 作者：Hang Wang, Chao Shen, Chenhao Lin, Minghui Yang, Lei Zhang, Cong Wang
🔗 链接：https://arxiv.org/abs/2605.00630 ；官方仓库 https://github.com/hwang-cs-ime/CMTA (空仓库)
📝 方法概述：识别 AIGC 视频特有的"跨模态时序指纹"——真实视频的视觉-文本对齐随语义自然波动，AIGV 在 prompt 约束下"过于稳定"。BLIP 给帧打 caption，CLIP 编码到共享空间；GRU 分支建模粗粒度跨模态稳定度，Transformer 分支建模细粒度帧间变化，融合分类。
💡 创新性分析：把"跨模态对齐稳态"作为 AIGV 指纹是新观察，与现有时空伪影路线正交；模块组合常规但概念贡献明确。
📊 关键指标：在 GenVideo / EvalCrafter / VideoPhy / VidProM 4 大数据集 / 40 子集上设新 SOTA，跨生成器泛化优势显著。**得分 81，已复现 (code/CMTA/)**

---

📄 标题：TabEmbed: Benchmarking and Learning Generalist Embeddings for Tabular Understanding
👥 作者：Minjie Qiang, Mingming Zhang, Xiaoyi Bao 等
🔗 链接：https://arxiv.org/abs/2605.04962 ；https://github.com/qiangminjie27/TabEmbed (评测代码已发；训练代码待录用后释放)
📝 方法概述：把 tabular 分类与检索统一为语义匹配任务，用大规模对比学习 + positive-aware 难负样本挖掘训练首个通用表格嵌入模型；同步推出 TabBench 评估基准。
💡 创新性分析：第一个面向表格的 generalist 嵌入工作，对 NLP foundation embedding 思路的迁移；TabBench 自身具社区价值。
📊 关键指标：在 TabBench 显著优于 SOTA 文本嵌入模型 (摘要未给百分点)。**得分 78**

---

📄 标题：Multimodal Data Curation Through Ranked Retrieval (SNS + EEE)
👥 作者：Pratyush Muthukumar, Harshil Kotamreddy, Sarah Amiraslani 等
🔗 链接：https://arxiv.org/abs/2605.01163
📝 方法概述：解决跨模态嵌入 modality gap + 监督噪声问题。Symmetric Nucleus Subsampling (SNS) 在训练对级修剪互相支撑的子片段；Expert Embedding Engine (EEE) 用学得投影融合多 embedding expert，并加 bias-aware 损失抑制 modality 偏置。
💡 创新性分析：同时干预"训练对"和"嵌入模型"两层，是合理的二阶设计；可作为大规模预训练数据 curator 插件。
📊 关键指标：modality gap 平均压缩 >90%；data blend 在下游模型上优于 stratified sampling 与传统 curation。**得分 77**

---

📄 标题：Beyond Semantic Similarity: Rethinking Retrieval for Agentic Search via Direct Corpus Interaction
👥 作者：Zhuofeng Li, Haoxiang Zhang, Cong Wei 等 (TIGER-Lab)
🔗 链接：https://arxiv.org/abs/2605.05242 ；https://huggingface.co/papers/2605.05242 (37 upvotes)
📝 方法概述：让 agent 直接用 grep / 文件读 / shell 等通用 unix 工具访问原始语料，**完全不走 embedding/向量索引/检索 API**。论文把现代检索抽象为"固定相似度接口下的 top-k 压缩"，认为对 agentic 任务这是瓶颈。
💡 创新性分析：在接口层挑战"是否需要 retriever"，与 ReAct/Toolformer 路线正交；工程门槛低，与 LLM agent 框架天然契合。
📊 关键指标：BRIGHT/BEIR 多个子集上明显超过 sparse/dense/rerank 强 baseline；BrowseComp-Plus 与 multi-hop QA 高准确率。**得分 75**

---

📄 标题：EKTM: Effective Knowledge Transfer for Multi-Task Recommendation Models
👥 作者：Guohao Cai, Jun Yuan, Zhenhua Dong
🔗 链接：https://arxiv.org/abs/2605.05730
📝 方法概述：CVR 标签稀疏问题。Router 模块跨任务聚合分发知识，每个 CVR 任务挂 transmitter 把 router 输出转换到本任务空间，再加 enhanced 模块抑制负迁移。
💡 创新性分析：相比 PLE / MMoE / STAR 是模块级增量；但工程动机和落地性强，正面解决 negative transfer。
📊 关键指标：商业 A/B eCPM +3.93%。**得分 71**

---

📄 标题：GRE-MC: Robust Multimodal Recommendation via Graph Retrieval-Enhanced Modality Completion
👥 作者：Yuan Li, Jun Hu, Jiaxin Jiang, Bryan Hooi, Bingsheng He
🔗 链接：https://arxiv.org/abs/2605.00670
📝 方法概述：处理多模态推荐数据的模态缺失。基于 modality-aware subgraph retrieval 在物品图上找语义相近邻居作为补全证据；Graph Transformer 全局注意力联合编码；learnable sparse-routing codebook 做正则。
💡 创新性分析：模态补全 + subgraph retrieval 的耦合是新组合；sparse routing codebook 类 VQ 思路稳定训练。
📊 关键指标：摘要未公布具体数据集与指标。**得分 69**

---

📄 标题：MiA-Signature: Approximating Global Activation for Long-Context Understanding
👥 作者：Yuqing Li, Jiangnan Li, Mo Yu, Zheng Lin, Weiping Wang, Jie Zhou
🔗 链接：https://arxiv.org/abs/2605.06416 ；https://huggingface.co/papers/2605.06416 (46 upvotes)
📝 方法概述：受认知科学"global ignition"启发，用子模 (submodular) 选择挑出能覆盖 query 激活上下文的高层概念集合作为压缩 signature，再用工作记忆轻量迭代精化，作为 RAG / agent 的条件信号。
💡 创新性分析：把"全局工作空间"概念形式化为 submodular cover 是较新颖灵感，与 RAG / agent 耦合方式具体可落。
📊 关键指标：多个 long-context understanding 任务一致提升 (摘要无具体数字)。**得分 69**

---

📄 标题：Negative Data Mining for Contrastive Learning in Dense Retrieval at IKEA.com
👥 作者：Eva Agapaki, Amritpal Singh Gill (IKEA.com)
🔗 链接：https://arxiv.org/abs/2605.00353
📝 方法概述：在 IKEA 双塔 late-interaction 检索上系统比较负采样策略。Structured negative sampling 利用商品分类树+属性挑难负；LLM-as-Judge 给所有候选商品打相关分，替代稀疏人工标注。
💡 创新性分析：方法层面是工程组合；论文价值在**诚实呈现 A/B 失利**——67% 热门 query 零点击率 >50%，归因到用户行为而非召回质量，对工业团队有实证启发。
📊 关键指标：离线类目准确率 +2.6%；在线 A/B 用户参与无显著差异 (p>0.05)。**得分 68**

---

📄 标题：Beyond Seeing Is Believing: On Crowdsourced Detection of Audiovisual Deepfakes
👥 作者：Michael Soprano, Andrea Cioci, Stefano Mizzaro
🔗 链接：https://arxiv.org/abs/2605.04797
📝 方法概述：Prolific 平台两组对照众包实验，96 视频×10 judgments=960 条人工判断，量化人类对真实/伪造、伪造类型、时间戳定位的准确率与一致性。
💡 创新性分析：方法层面是经典 user study；价值在为 governance pipeline 提供实证基线，提示自动化系统应聚焦"定位"。
📊 关键指标：authenticity 二分类聚合后稳定，但 modality attribution 噪声高，audio+video 联合伪造识别尤难。**得分 44**
