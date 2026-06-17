# LLM Judges Have Dark Current: Psychometric Datasheet for LLM-as-a-Judge

## 基本信息 / Basic Info

| 字段 | 内容 |
|------|------|
| **标题** | LLM Judges Have Dark Current: A Psychometric Datasheet for LLM-as-a-Judge Evaluation |
| **作者** | Hiroyasu Usami, Keisuke Hara, Ayato Tsuboi, Naohiko Matsuda |
| **机构** | (未公开机构信息) |
| **链接** | https://arxiv.org/abs/2606.15610 |
| **arXiv ID** | 2606.15610 |
| **提交日期** | June 14, 2026 ★ (在June 16 Monday arXiv listing中) |
| **Bucket** | WEAK |

---

## 方法概述 / Method Overview

**中文：**  
LLM-as-a-judge（LLM作为评分器）已广泛用于模型评估和数据标注，但缺乏系统性质量度量。本文引入**Judge Datasheet协议**，从心理测量学视角对LLM评判器进行系统审计，揭示五类偏差：①暗电流（Dark Current，在无效输入下的虚假判断）；②跨敏感性（Cross-Sensitivity，对同质量变体的不稳定判断）；③位置偏好（Positional False Preference）；④目标敏感性（Target Sensitivity，质量梯度上的区分能力）；⑤判断准则偏移。通过对Llama-3.1-8B、Qwen2.5-14B、Qwen2.5-32B三个开源评判模型的案例研究验证。

**English:**  
This paper introduces a Judge Datasheet protocol that treats LLM judges as psychometric measurement instruments. It audits five bias dimensions: dark current (spurious judgments on null inputs), cross-sensitivity (instability on same-quality variants), positional false preference, target sensitivity (ability to rank on a quality ladder), and tie-instruction-induced operating-point shift. Validated on three open-weight judges (Llama-3.1-8B, Qwen2.5-14B, Qwen2.5-32B).

---

## 故事弧线 / Story Arc

**现有方法不足 →** LLM评判器作为数据标注工具被广泛使用，但质量评估缺乏标准化，各种偏差（位置偏好、格式偏好等）未被系统量化。  
**本文设计 →** 借鉴心理测量学建立Judge Datasheet标准协议，五维度量化评判器的系统性偏差，指导评判器选择和使用。

---

## 创新性 / Innovation

1. **心理测量学框架迁移**：将心理测量学（Psychometrics）引入LLM评判器审计，方法论创新。
2. **暗电流概念**：类比物理/电子领域的暗电流，描述LLM评判器在无有效信号时的本底噪声判断。
3. **五维度系统量化**：首次提供可复现的标准化评判器质量报告协议。

**与前工作的差异：** 现有LLM-as-a-judge研究多关注准确率，忽视系统性偏差；本文提供更细粒度的偏差剖析。

---

## 评分 / Scoring

| 维度 | 分数 | 满分 | 说明 |
|------|------|------|------|
| 创新性 (Innovation) | 18 | 30 | 框架新颖但属方法论贡献 |
| 实验SOTA增量 (SOTA Delta) | 9 | 15 | 诊断性研究，无直接SOTA对比 |
| 实验质量/消融 (Exp Quality) | 10 | 15 | 三模型案例研究充分 |
| 效率 (Efficiency) | 7 | 10 | 审计协议本身轻量 |
| 泛化性 (Generalization) | 4 | 5 | 可普遍应用于任何LLM评判器 |
| 领域相关性 (Domain Relevance) | 18 | 25 | 与数据标注质量、内容评审系统直接相关 |
| **Total** | **66** | **100** | |

---

## 参考链接

- arXiv: https://arxiv.org/abs/2606.15610
