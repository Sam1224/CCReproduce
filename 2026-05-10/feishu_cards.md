# Feishu Cards — 2026-05-10 Daily AI Paper Inspection

> Papers with score ≥ 40. All cards in Chinese.

---

📄 标题：EKTM: Effective Knowledge Transfer for Multi-Task Recommendation Models
👥 作者：Guohao Cai, Jun Yuan, Zhenhua Dong
🔗 链接：https://arxiv.org/abs/2605.05730
📝 方法概述：电商转化率（CVR）模型因转化行为稀疏而面临数据不足挑战。EKTM 提出 Router-Transmitter-Enhanced 三模块架构：Router 整合并动态分发跨任务知识，每个 CVR 任务独立配备 Transmitter 将 Router 知识适配至本任务，Enhanced 模块保护原任务学习不被干扰。在多基准数据集上超越现有 SOTA，商业平台在线 A/B 测试验证并已全量部署于两个主流量场景。
💡 创新性分析：相比 MMOE/PLE 等经典多任务架构，EKTM 专注于 CVR 稀疏场景下的跨任务知识转化路径而非仅共享底层表示；Router 的动态路由机制有效缓解负迁移；Enhanced 模块的保护设计是精细工程创新。工业 A/B 测试是学术界罕见的强验证。
📊 关键指标：商业平台在线 A/B 测试 eCPM 提升 +3.93%；两个主流量场景全量部署；多基准数据集达到 SOTA（AUC/GAUC）。综合评分 82/100。

---

📄 标题：AFMRL: Attribute-Enhanced Fine-Grained Multi-Modal Representation Learning in E-commerce
👥 作者：Biao Zhang, Lixin Chen, Bin Zhang, Zongwei Wang, Tong Liu, Bo Zheng（阿里巴巴淘宝天猫）
🔗 链接：https://arxiv.org/abs/2604.20135
📝 方法概述：现有多模态表示模型（如 VLM2Vec）缺乏细粒度语义辨别力，无法有效区分相似商品。AFMRL 将细粒度商品理解重构为属性生成任务，通过两阶段框架：①属性引导对比学习（AGCL）——利用 MLLM 提取的关键属性精准识别难负样本和过滤假负样本；②检索感知属性强化（RAR）——以检索性能提升为奖励信号反向强化属性生成，形成生成-检索正向飞轮。在阿里巴巴大规模电商数据集上多下游检索任务达到 SOTA。
💡 创新性分析：将表示学习问题转化为生成任务是创新视角，使 MLLM 生成能力直接服务于表示质量；属性引导的难负样本挖掘比随机采样更精准；RAR 的奖励机制使属性生成和检索性能形成正向飞轮。来自阿里淘宝最大电商平台，结果可信度高。
📊 关键指标：淘宝大规模电商数据集多任务检索 SOTA（Recall@1、MRR、nDCG@10），超越 VLM2Vec 等基线。综合评分 81/100。

---

📄 标题：LatentRAG: Latent Reasoning and Retrieval for Efficient Agentic RAG
👥 作者：（未完全披露）
🔗 链接：https://arxiv.org/abs/2605.06285
📝 方法概述：Agentic RAG 通过多步推理解决复杂问题，但自回归生成中间思路和子查询带来高延迟。LatentRAG 将推理和检索从离散语言空间迁移至连续隐空间：LLM 直接从隐状态生成隐向量 tokens 作为思路和子查询，通过隐空间将 LLM 与密集检索器对齐，支持端到端联合优化。在 7 个基准数据集上，准确率与显式 Agentic RAG 相当，推理延迟降低约 90%。
💡 创新性分析：将 LLM 推理迁移至连续空间是根本性创新，突破了自回归延迟瓶颈；LLM-检索器隐空间联合对齐是技术难点突破；单次前向传递消除多次自回归调用；端到端优化允许检索和推理共同训练。
📊 关键指标：7个多跳QA基准准确率与显式AgRAG持平，推理延迟降低约 90%，接近单步RAG延迟水平。综合评分 78/100。

---

📄 标题：LiSCP: Lightweight Stylistic Consistency Profiling for Multimedia Moderation
👥 作者：Siyuan Li 等（9位作者）
🔗 链接：https://arxiv.org/abs/2605.05950
📝 方法概述：区分人类撰写与 LLM 生成内容是多媒体内容审核的关键任务，现有检测器在对抗性改写下鲁棒性差。LiSCP 构建融合离散文体特征和连续语义信号的一致性档案（consistency profile），通过多模态引导的改写变体检测特征稳定性：LLM 生成文本在改写下文体保持稳定，而人类写作文体变化大，利用此差异实现对改写攻击鲁棒的检测，且整体轻量适合部署。
💡 创新性分析：基于改写稳定性检测的范式是对对抗攻击的根本性防御；多模态引导改写增加跨模态对抗鲁棒性；离散+连续特征融合比单一特征更全面；轻量设计面向实际多媒体审核管线。
📊 关键指标：多媒体内容审核基准对抗改写场景下鲁棒性显著优于基线；无攻击场景下保持高准确率；轻量实现适合生产部署。综合评分 77/100。

---

📄 标题：VANGUARD: Reasoning-Guided Grounding for Video Anomaly Detection via MLLMs
👥 作者：Sakshi Agarwal, Aishik Konwer, Ankit Parag Shah
🔗 链接：https://arxiv.org/abs/2605.02912
📝 方法概述：视频异常检测传统上缺乏可解释推理和精确空间定位。VANGUARD 将异常分类、空间定位与链式推理统一在单一 VLM 中，通过三阶段课程式训练逐步叠加：①分类热身（冻结 backbone）；②LoRA 空间定位适配；③链式思维生成。在 UCF-Crime 基准达到 94% ROC-AUC + 84% F1，同时提供可解释 CoT 推理和 bounding box 定位。
💡 创新性分析：分类+定位+推理三能力在单一VLM中统一是首创；三阶段课程训练避免任务干扰；VAD+CoT可解释性是新颖结合；LoRA适配专门解决VLM定位幻觉问题。可迁移至直播内容异常检测（违禁商品展示、违规行为识别）。
📊 关键指标：UCF-Crime ROC-AUC 94%（基线约91%），F1 84%（基线约78%），同时具备空间定位和CoT推理能力。综合评分 71/100。

---

📄 标题：PREFER: Personalized Review Summarization with Online Preference Learning
👥 作者：Millend Roy, Agostino Capponi, Vineet Goyal（哥伦比亚大学）
🔗 链接：https://arxiv.org/abs/2605.05911
📝 方法概述：电商平台海量评论导致信息过载，现有摘要系统生成通用静态摘要无法匹配个体偏好。PREFER 提出在线偏好学习框架：系统通过已生成摘要中直接获取的用户反馈迭代精化对用户偏好的理解，为每位用户生成针对其关注维度（耐用性、外观、性价比等）的定制化摘要，框架支持偏好的时变动态更新。在 Amazon Reviews'23 数据集受控仿真中验证对齐度显著提升。
💡 创新性分析：在线学习范式从离线偏好建模转向在线迭代精化，更接近实际部署；从摘要本身获取反馈是独特信号来源；动态偏好表示比静态个性化更贴近用户行为规律。
📊 关键指标：Amazon Reviews'23 受控仿真，在线偏好学习对齐目标用户兴趣显著提升，摘要质量（ROUGE等）保持稳定。综合评分 68/100。

---

📄 标题：Who Decides What Is Harmful? Multi-Agent Personalised Content Moderation
👥 作者：Ewelina Gajewska, Michal Wawer, Katarzyna Budzynska, Jaroslaw A. Chudziak
🔗 链接：https://arxiv.org/abs/2605.01416
📝 方法概述：传统内容审核的中心化规则无法适应有害内容感知的主观差异。本文提出基于 LLM 的多智能体个性化推断框架：Expert Agents 处理不同类型有害内容，Manager Agent 协调分析流程和智能体选择，Ghost Profile Agent 模拟用户视角将用户敏感度档案转化为决策输入。与一系列非个性化基线相比准确率提升高达 32%，表现出更强的用户敏感度对齐能力。
💡 创新性分析：从平台级统一规则转向用户级个性化审核是有价值研究方向；Ghost Profile Agent 将敏感度档案转化为 LLM 可推理视角设计巧妙；多智能体协作架构在可扩展性上有优势。可应用于达人（KOL）内容合规管理场景。
📊 关键指标：与多种非个性化基线相比准确率提升高达 32%，用户敏感度对齐度显著更优。综合评分 64/100。

---

📄 标题：EGAD: Entropy-Guided Adaptive Distillation for Token-Level Knowledge Transfer
👥 作者：Hao Zhang, Zhibin Zhang, Guangxin Wu, Wanyi Ning, Jiafeng Guo, Xueqi Cheng（中国科学院计算技术研究所）
🔗 链接：https://arxiv.org/abs/2605.01732
📝 方法概述：现有 LLM 蒸馏方法对所有 token 平等对待，忽视不同 token 对模型决策贡献度的差异。EGAD 提出基于 token 级熵的自适应蒸馏策略：①Token 级课程学习（从低熵到高熵渐进训练）；②自适应蒸馏温度（根据教师熵动态调整）；③双分支架构（低熵 token 用 logits-only 蒸馏，高熵 token 用特征级深度蒸馏）。
💡 创新性分析：将教师输出不确定性转化为蒸馏强度信号，直觉合理且实现清晰；三个机制协同设计，各有独立动机；双分支架构兼顾效率与效果。可应用于电商场景下的轻量模型压缩。
📊 关键指标：标准LLM蒸馏基准上超越传统蒸馏方法；双分支架构降低低熵token计算开销。综合评分 61/100。
