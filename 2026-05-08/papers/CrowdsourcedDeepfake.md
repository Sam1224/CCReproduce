# Beyond Seeing Is Believing: On Crowdsourced Detection of Audiovisual Deepfakes

- **arXiv**: https://arxiv.org/abs/2605.04797
- **作者**: Michael Soprano, Andrea Cioci, Stefano Mizzaro
- **领域标签**: `Deepfake` `Crowdsourcing` `Content Governance` `User Study`

## 方法概述
不是新算法，而是**人因实验**：在 Prolific 平台用 96 个视频 (AV-Deepfake1M / TMC 各 48) 收集 960 条工人判断 (每视频 10 条)，量化人类对 (1) 真实/伪造、(2) 伪造类型 (audio-only / video-only / audio-video) 、(3) 伪造时间戳定位的准确率与一致性。

## 故事
Deepfake 真伪难辨 → 业内默认依赖人工 review → 但人到底能识别到什么程度？没有系统量化 → 做对照众包实验回答这个问题。

## 创新性分析
- 方法学是经典众包实验设计，**算法层面无新点**；
- 但作为 governance pipeline 的实证数据点很有用：人工 review 能稳定判断 authenticity，但定位 modality 和时间戳很差，提示自动化系统应聚焦"定位"而非"二分类"。

## 关键指标
- 工人极少把真实视频误判为伪造；
- 多次 judgment 聚合可稳定 authenticity 信号；
- modality attribution 噪声远高于 authenticity；audio+video 联合伪造识别尤难。

## 评分
| 维度 | 分 / 满分 | 理由 |
|------|----------|------|
| 创新性 | 10 / 30 | 标准 user study |
| 实验指标 | 5 / 15 | 无算法 SOTA |
| 实验质量 | 10 / 15 | 实验设计扎实 (96 vid × 10 judgments) |
| 效率 | 4 / 10 | 不适用 |
| 泛化性 | 3 / 5 | 两个数据集 |
| 相关性 | 12 / 25 | 对达人内容审核团队工作流有参考意义 |
| **合计** | **44** |
